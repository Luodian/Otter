import mimetypes
import os
from io import BytesIO
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
import sys

sys.path.append("/mnt/petrelfs/zhangyuanhan/Otter/")
from src.otter_ai.models.otter.modeling_otter import OtterForConditionalGeneration
from .base_model import BaseModel

# Disable warnings
requests.packages.urllib3.disable_warnings()


class OtterVideo(BaseModel):
    def __init__(self, model_path="luodian/OTTER-Video-LLaMA7B-DenseCaption", load_bit="bf16"):
        super().__init__("otter_video", model_path)
        precision = {}
        if load_bit == "bf16":
            precision["torch_dtype"] = torch.bfloat16
        elif load_bit == "fp16":
            precision["torch_dtype"] = torch.float16
        elif load_bit == "fp32":
            precision["torch_dtype"] = torch.float32
        self.model = OtterForConditionalGeneration.from_pretrained(model_path, device_map="sequential", **precision)
        self.tensor_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[load_bit]
        self.model.text_tokenizer.padding_side = "left"
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = transformers.CLIPImageProcessor()
        self.model.eval()

    def get_formatted_prompt(self, prompt: str) -> str:
        return f"<image>User: {prompt} GPT:<answer>"

    def extract_frames(self, video_path, num_frames=16):
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = total_frames // num_frames
        frames = []

        for i in range(num_frames):
            video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame).convert("RGB")
                frames.append(frame)

        video.release()
        return frames

    def get_response(
        self,
        input_data,
        prompt: str,
        model=None,
        image_processor=None,
        tensor_dtype=None,
    ) -> str:
        if isinstance(input_data, Image.Image):
            vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        elif isinstance(input_data, list):  # list of video frames
            vision_x = image_processor.preprocess(input_data, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")

        lang_x = self.tokenizer(
            [
                self.get_formatted_prompt(prompt),
            ],
            return_tensors="pt",
        )

        bad_words_id = self.tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        # import pdb;pdb.set_trace()
        generated_text = self.model.generate(
            vision_x=vision_x.to(model.device, dtype=tensor_dtype),
            lang_x=lang_x["input_ids"].to(model.device),
            attention_mask=lang_x["attention_mask"].to(model.device),
            max_new_tokens=512,
            num_beams=3,
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words_id,
        )
        parsed_output = model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
        return parsed_output

    def generate(self, input_data):
        video_dir = input_data.get("video_root", "")
        frames_list = self.extract_frames(input_data["video_path"])

        object_description = input_data["object_description"]

        if object_description != "None":
            context = f"Given context:{object_description}. "
        else:
            context = ""
        prompts_input = context + input_data["question"]

        response = self.get_response(
            frames_list,
            prompts_input,
            self.model,
            self.image_processor,
            self.tensor_dtype,
        )

        return response


if __name__ == "__main__":
    pass
