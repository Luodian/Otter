from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
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


class InstructBLIP(BaseModel):
    def __init__(self, model_path: str = "Salesforce/instructblip-vicuna-7b", cuda_id: int = 0, max_new_tokens=32):
        super().__init__("instructblip", model_path)
        self.device = f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.processor = InstructBlipProcessor.from_pretrained(model_path)
        self.max_new_tokens = max_new_tokens

    def generate(self, text_prompt: str, raw_image_data: str):
        raw_image_data = get_pil_image(raw_image_data)
        raw_image_data = raw_image_data.convert("RGB")
        formatted_prompt = f"{text_prompt}\nAnswer:"
        # Accordling to https://huggingface.co/Salesforce/instructblip-vicuna-7b . Seems that is is no special prompt format for instruct blip
        model_inputs = self.processor(images=raw_image_data, text=formatted_prompt, return_tensors="pt").to(self.device)
        # We follow the recommended parameter here:https://huggingface.co/Salesforce/instructblip-vicuna-7b
        generation_output = self.model.generate(**model_inputs, do_sample=False, max_new_tokens=self.max_new_tokens, min_length=1)
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
        return generation_text[0]

    def eval_forward(self, question, answer, image):
        raise NotImplementedError
