import json

import clip
import ffmpeg
import numpy as np
import torch
from transformers import DebertaV2Tokenizer, DebertaV2Config

from .FrozenBiLM.model import DebertaV2ForMaskedLM
from .FrozenBiLM.extract.preprocessing import Preprocessing
from .FrozenBiLM.util.misc import get_mask
from .base_model import BaseModel


class FrozenBilm(BaseModel):
    def __init__(self, model_path: str, vocab_path: str, vit_download_path: str):
        # In the official repo, they use msrvtt vocab
        self.vocab = json.load(open(vocab_path, "r"))
        self.id2a = {y: x for x, y in self.vocab.items()}

        self.max_feats = 10
        self.features_dim = 768
        config = DebertaV2Config.from_pretrained(pretrained_model_name_or_path="microsoft/deberta-v2-xlarge")
        n_ans = len(self.vocab)
        # Default setting for the official checkpoints
        self.model = DebertaV2ForMaskedLM(
            config=config,
            max_feats=10,
            features_dim=768,
            freeze_lm=True,
            freeze_mlm=True,
            ds_factor_attn=8,
            ds_factor_ff=8,
            ft_ln=True,
            dropout=0.1,
            n_ans=n_ans,
            freeze_last=True,
        )
        self.model.eval()
        # print("loading from : ", model_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"], strict=False)

        self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge")

        # Init answer embedding module
        # Setting 5 here is because the default setting of the
        # max_atokens is 5 in the repo
        aid2tokid = torch.zeros(len(self.vocab), 5).long()
        for a, aid in self.vocab.items():
            tok = torch.tensor(
                self.tokenizer(
                    a,
                    add_special_tokens=False,
                    max_length=5,
                    truncation=True,
                    padding="max_length",
                )["input_ids"],
                dtype=torch.long,
            )
            aid2tokid[aid] = tok
        self.model.set_answer_embeddings(aid2tokid.to(self.model.device), freeze_last=False)

        # print("loading visual backbone")
        self.preprocess = Preprocessing()
        self.backbone, _ = clip.load("ViT-L/14", download_root=vit_download_path, device=self.model.device)
        self.backbone.eval()
        super().__init__(model_name="frozen_bilm", model_path=model_path)

    def generate(self, input_data: dict):
        video_path = input_data["video_path"]
        # Extract frames from video
        # print("extracting visual features")
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        num, denum = video_stream["avg_frame_rate"].split("/")
        frame_rate = int(num) / int(denum)
        if height >= width:
            h, w = int(height * 224 / width), 224
        else:
            h, w = 224, int(width * 224 / height)
        assert frame_rate >= 1

        cmd = ffmpeg.input(video_path).filter("fps", fps=1).filter("scale", w, h)
        x = int((w - 224) / 2.0)
        y = int((h - 224) / 2.0)
        cmd = cmd.crop(x, y, 224, 224)
        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(capture_stdout=True, quiet=True)

        h, w = 224, 224
        video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        video = torch.from_numpy(video.astype("float32"))
        video = video.permute(0, 3, 1, 2)
        video = video.squeeze()
        video = self.preprocess(video)
        # Remove video to device so that we don't encounter OOM
        video = self.backbone.encode_image(video)

        # Subsample or pad
        if len(video) >= self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        else:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, 768).to(self.model.device)], 0)
        video = video.unsqueeze(0).to(self.model.device)
        video_mask = get_mask(torch.tensor(video_len, dtype=torch.long).unsqueeze(0), video.size(1)).to(self.model.device)
        # print("visual features extracted")

        # Process question
        question = input_data["prompt"].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        suffix = "."

        text = f"Question: {question} Answer: {self.tokenizer.mask_token}{suffix}"

        if input_data["task"] == "Generation":
            encoded = self.tokenizer(
                [text],
                add_special_tokens=True,
                max_length=512,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(self.model.device)
            attention_mask = encoded["attention_mask"].to(self.model.device)

            # remove sep token if not using the suffix
            # We don't use suffix here so we just run this
            if not suffix:
                attention_mask[input_ids == self.tokenizer.sep_token_id] = 0
                input_ids[input_ids == self.tokenizer.sep_token_id] = self.tokenizer.pad_token_id
            # print("encoded text")

            output = self.model(
                video=video,
                video_mask=video_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = output["logits"]
            delay = self.max_feats
            logits = logits[:, delay : encoded["input_ids"].size(1) + delay][encoded["input_ids"] == self.tokenizer.mask_token_id]  # get the prediction on the mask token
            logits = logits.softmax(-1)
            topk = torch.topk(logits, 5, -1)
            topk_txt = [[self.id2a[x.item()] for x in y] for y in topk.indices.cpu()]
            topk_scores = [[f"{x:.2f}".format() for x in y] for y in topk.values.cpu()]
            # print(topk_scores)
            topk_all = [[x + "(" + y + ")" for x, y in zip(a, b)] for a, b in zip(topk_txt, topk_scores)]
            result = topk_txt[0][0]
            # print(f"Top 5 answers and scores: {topk_all[0]}")
        else:
            logits_list = []
            for cur_option in input_data["option"]:
                text = f"Question: {question} Is it '{cur_option}'? {self.tokenizer.mask_token}{suffix}"
                encoded = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=512,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                )
                # forward
                output = self.model(
                    video=video,
                    video_mask=video_mask,
                    input_ids=encoded["input_ids"].to(self.model.device),
                    attention_mask=encoded["attention_mask"].to(self.model.device),
                )
                logits = output["logits"]
                # get logits for the mask token
                delay = 10
                logits = logits[:, delay : encoded["input_ids"].size(1) + delay][encoded["input_ids"] == self.tokenizer.mask_token_id]
                logits_list.append(logits.softmax(-1)[:, 0])
            logits = torch.stack(logits_list, 1)
            if logits.shape[1] == 1:
                preds = logits.round().long().squeeze(1)
            else:
                preds = logits.max(1).indices
            # print(preds)
            choices = ["A", "B", "C", "D"]
            result = choices[int(preds)]
        print(f"Question : {input_data['prompt']}")
        print(f"Result : {result}")
        return result

    def to(self, device):
        device = torch.device(device)
        self.model = self.model.to(device)
        # self.backbone = self.backbone.to(device)


if __name__ == "__main__":
    model = FrozenBilm("/mnt/lustre/yhzhang/kaichen/frozenbilm/frozenbilm.pth", vocab_path="/mnt/lustre/yhzhang/kaichen/frozenbilm/vocab.json", vit_download_path="/mnt/lustre/yhzhang/kaichen/frozenbilm/")
    device = torch.device("cpu")
    model.to(device)
    data = {"video_idx": "./data_source/multi_hop_reasoning/03f2ed96-1719-427d-acf4-8bf504f1d66d.mp4", "question": "What is in this image?"}
    print(model.generate(data))
