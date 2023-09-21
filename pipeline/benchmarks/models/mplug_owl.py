import os

import torch
from transformers import AutoTokenizer
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import (
    MplugOwlImageProcessor,
    MplugOwlProcessor,
)

from .base_model import BaseModel

pretrained_ckpt = "MAGAer13/mplug-owl-llama-7b-video"


class mPlug_owl(BaseModel):
    def __init__(self, model_path: str):
        super().__init__("mplug_owl", model_path)
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            pretrained_ckpt,
            torch_dtype=torch.bfloat16,
        )
        self.image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.model.eval()

    def format_prompt(self, question):
        prompts = [f" <|video|> Question : {question} Answer : "]
        return prompts

    def generate(self, input_data: dict):
        questions = input_data["question"]
        video_dir = input_data.get("video_root", "")
        video_list = input_data["video_path"]
        generate_kwargs = {"do_sample": True, "top_k": 5, "max_length": 512}

        object_description = input_data["object_description"]
        if object_description != "None":
            context = f"Given context:{object_description}. "
        else:
            context = ""
        prompts_input = context + input_data["question"]

        prompt = self.format_prompt(prompts_input)
        inputs = self.processor(text=prompt, videos=video_list, num_frames=4, return_tensors="pt")
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        sentence = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        return sentence


if __name__ == "__main__":
    model = mPlug_owl("")
    device = torch.device("cuda")
    model.model = model.model.to(device)
    data = {
        "video_idx": ["./data_source/multi_hop_reasoning/03f2ed96-1719-427d-acf4-8bf504f1d66d.mp4"],
        "question": "What is in this image?",
    }
    print(model.generate(data))
