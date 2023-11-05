from transformers import FuyuForCausalLM, AutoTokenizer, FuyuImageProcessor, FuyuProcessor
from PIL import Image
from .base_model import BaseModel
import torch
import numpy as np
import warnings
import io
import base64

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


import math


class OtterHD(BaseModel):
    def __init__(self, model_path: str = "Otter-AI/OtterHD-8B", cuda_id: int = 0, resolution: int = -1, max_new_tokens=256):
        super().__init__("otterhd", model_path)
        self.resolution = resolution
        self.device = f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
        self.model = FuyuForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(image_processor=self.image_processor, tokenizer=self.tokenizer)
        self.max_new_tokens = max_new_tokens

    def generate(self, text_prompt: str, raw_image_data: str):
        raw_image_data = get_pil_image(raw_image_data)
        # make sure the image is in RGB format and resize to match the width
        raw_image_data = raw_image_data.convert("RGB")
        if self.resolution != -1:
            width, height = raw_image_data.size
            short_edge = min(width, height)
            scaling_factor = self.resolution / short_edge
            new_width = math.ceil(width * scaling_factor)
            new_height = math.ceil(height * scaling_factor)
            raw_image_data = raw_image_data.resize((new_width, new_height), Image.ANTIALIAS)

        formated_prompt = f"User: {text_prompt} Assistant:"
        model_inputs = self.processor(text=formated_prompt, images=[raw_image_data], device=self.device)
        for k, v in model_inputs.items():
            model_inputs[k] = v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else [vv.to(self.device, non_blocking=True) for vv in v]

        model_inputs["image_patches"][0] = model_inputs["image_patches"][0].to(dtype=next(self.model.parameters()).dtype)
        generation_output = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
        response = generation_text[0].split("\x04")[1].strip(" ").strip("\n")
        return response

    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for Fuyu
        pass
