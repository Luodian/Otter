from typing import List
from transformers import AutoTokenizer, FuyuImageProcessor
from transformers import FuyuForCausalLM
from src.otter_ai.models.fuyu.processing_fuyu import FuyuProcessor
from PIL import Image
from .base_model import BaseModel
import torch
import numpy as np
import warnings
import io
import base64
import math

warnings.filterwarnings("ignore")


def get_pil_image(raw_image_data) -> Image.Image:
    if isinstance(raw_image_data, Image.Image):
        return raw_image_data

    elif isinstance(raw_image_data, dict) and "bytes" in raw_image_data:
        return Image.open(io.BytesIO(raw_image_data["bytes"]))

    elif isinstance(raw_image_data, str):  # Assuming this is a base64 encoded string
        image_bytes = base64.b64decode(raw_image_data)
        return Image.open(io.BytesIO(image_bytes))

    else:
        raise ValueError("Unsupported image data format")


class Fuyu(BaseModel):
    def __init__(self, model_path: str = "adept/fuyu-8b", cuda_id: int = 0, resolution: int = -1, max_new_tokens=256):
        super().__init__("fuyu", model_path)
        self.resolution = resolution
        self.device = f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
        self.model = FuyuForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("adept/fuyu-8b")
        self.image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(image_processor=self.image_processor, tokenizer=self.tokenizer)
        self.max_new_tokens = max_new_tokens
        self.bad_words_list = ["User:", "Assistant:"]
        self.bad_words_ids = self.tokenizer(self.bad_words_list, add_special_tokens=False).input_ids

    def generate(self, text_prompt: str, raw_image_data: str):
        raw_image_data = get_pil_image(raw_image_data)
        raw_image_data = raw_image_data.convert("RGB")
        # make sure the image is in RGB format and resize to match the width
        if self.resolution != -1:
            width, height = raw_image_data.size
            short_edge = min(width, height)
            scaling_factor = self.resolution / short_edge
            new_width = math.ceil(width * scaling_factor)
            new_height = math.ceil(height * scaling_factor)
            raw_image_data = raw_image_data.resize((new_width, new_height), Image.ANTIALIAS)
        # formated_prompt = f"User: {text_prompt} Assistant:"
        model_inputs = self.processor(text=text_prompt, images=[raw_image_data], device=self.device)
        for k, v in model_inputs.items():
            model_inputs[k] = v.to(self.device)

        model_inputs["image_patches"] = model_inputs["image_patches"].to(dtype=next(self.model.parameters()).dtype)
        generation_output = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id, bad_words_ids=self.bad_words_ids)
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
        return generation_text[0].split("\x04")[1].strip(" ").strip("\n")

    def eval_forward(self, **kwargs):
        return super().eval_forward(**kwargs)


if __name__ == "__main__":
    model = Fuyu()
    print(model.generate("Generate a coco-style caption.\n", Image.open("/home/luodian/projects/Otter/archived/test_images/rabbit.png").convert("RGB")))
