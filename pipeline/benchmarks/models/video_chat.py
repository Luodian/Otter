from .base_model import BaseModel
from .Ask_Anything.video_chat.utils.config import Config
from .Ask_Anything.video_chat.models.videochat import VideoChat as VideoChatModel
from .Ask_Anything.video_chat.utils.easydict import EasyDict
from .Ask_Anything.video_chat.models.video_transformers import (
    GroupNormalize,
    GroupScale,
    GroupCenterCrop,
    Stack,
    ToTorchFormatTensor,
)

import os
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

config_file = "/mnt/petrelfs/zhangyuanhan/Otter/pipeline/evaluation/models/Ask_Anything/video_chat/configs/config.json"
cfg = Config.from_file(config_file)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False


class VideoChat(BaseModel):
    # checkpoint will be automatically downloaded
    def __init__(self, model_path: str):
        super().__init__("video_chat", model_path)
        self.model = VideoChatModel(config=cfg.model)

        self.model = self.model.to(torch.device(cfg.device))
        self.model = self.model.eval()

        self.chat = EasyDict(
            {
                #     "system": "You are an AI assistant. A human gives an image or a video and asks some questions. You should give helpful, detailed, and polite answers.\n",
                "system": "",
                "roles": ("Human", "Assistant"),
                "messages": [],
                "sep": "###",
            }
        )

    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([start + int(np.round(seg_size * idx)) for idx in range(num_segments)])
        return offsets

    def load_video(self, video_path, num_segments=8, return_msg=False):
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        frame_indices = self.get_index(num_frames, num_segments)

        # transform
        crop_size = 224
        scale_size = 224
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]

        transform = T.Compose(
            [
                GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
                GroupCenterCrop(crop_size),
                Stack(),
                ToTorchFormatTensor(),
                GroupNormalize(input_mean, input_std),
            ]
        )

        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        torch_imgs = transform(images_group)
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return torch_imgs, msg
        else:
            return torch_imgs

    def get_prompt(self, conv):
        ret = conv.system + conv.sep
        for role, message in conv.messages:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
        return ret

    def get_context_emb(self, conv, model, img_list):
        prompt = self.get_prompt(conv)
        print(prompt)
        if "<VideoHere>" in prompt:
            prompt_segs = prompt.split("<VideoHere>")
        else:
            prompt_segs = prompt.split("<ImageHere>")
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            model.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def answer(
        self,
        conv,
        model,
        img_list,
        max_new_tokens=200,
        num_beams=1,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        temperature=1.0,
    ):
        stop_words_ids = [
            torch.tensor([835]).to("cuda:0"),
            torch.tensor([2277, 29937]).to("cuda:0"),
        ]  # '###' can be encoded in two different ways.
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        conv.messages.append([conv.roles[1], None])
        embs = self.get_context_emb(conv, model, img_list)
        outputs = model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split("###")[0]  # remove the stop sign '###'
        output_text = output_text.split("Assistant:")[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def generate(self, input_data):
        inputs = {}
        video_dir = input_data.get("video_root", "")
        vid, msg = self.load_video(input_data["video_path"], num_segments=8, return_msg=True)
        # print(msg)
        object_description = input_data["object_description"]
        if object_description != "None":
            context = f"Given context:{object_description}. "
        else:
            context = ""
        prompts_input = context + input_data["question"]

        self.chat.messages.append([self.chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
        self.chat.messages.append([self.chat.roles[0], prompts_input + "\n"])

        # The model expects inputs of shape: T x C x H x W
        TC, H, W = vid.shape
        video = vid.reshape(1, TC // 3, 3, H, W).to(self.model.device)
        img_list = []
        image_emb, _ = self.model.encode_img(video)
        img_list.append(image_emb)

        result = self.answer(conv=self.chat, model=self.model, img_list=img_list, max_new_tokens=1000)[0]
        self.chat.messages = []
        return result


if __name__ == "__main__":
    model = VideoChat("")
    data = {
        "video_idx": "03f2ed96-1719-427d-acf4-8bf504f1d66d.mp4",
        "question": "What is in this image?",
    }
    print(model.generate(data))
