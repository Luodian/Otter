import mimetypes
import os
from io import BytesIO
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image

from otter_ai import OtterForConditionalGeneration
from .model import Model


# Disable warnings
requests.packages.urllib3.disable_warnings()


class Otter(Model):
    def __init__(self, model_path="luodian/OTTER-Image-MPT7B", load_bit="bf16"):
        super().__init__("otter", model_path)
        precision = {}
        if load_bit == "bf16":
            precision["torch_dtype"] = torch.bfloat16
        elif load_bit == "fp16":
            precision["torch_dtype"] = torch.float16
        elif load_bit == "fp32":
            precision["torch_dtype"] = torch.float32
        self.model = OtterForConditionalGeneration.from_pretrained(model_path, device_map="sequential", **precision)
        self.model.text_tokenizer.padding_side = "left"
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = transformers.CLIPImageProcessor()
        self.model.eval()

    def get_formatted_prompt(prompt: str) -> str:
        return f"<image>User: {prompt} GPT:<answer>"

    def generate(self, question: str, raw_image_data: Image.Image, no_image_flag=False):
        input_data = image
        if isinstance(input_data, Image.Image):
            if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
                vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(self.model.parameters()).dtype)
            else:
                vision_x = self.image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image.")

        lang_x = self.model.text_tokenizer(
            [
                self.get_formatted_prompt(question),
            ],
            return_tensors="pt",
        )

        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)
        lang_x_input_ids = lang_x["input_ids"]
        lang_x_attention_mask = lang_x["attention_mask"]

        generated_text = model.generate(
            vision_x=vision_x.to(model.device),
            lang_x=lang_x_input_ids.to(model.device),
            attention_mask=lang_x_attention_mask.to(model.device),
            max_new_tokens=512,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        parsed_output = model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].split("<|endofchunk|>")[0].strip()
        return parsed_output


if __name__ == "__main__":
    model = Otter("/data/pufanyi/training_data/checkpoints/OTTER-Image-MPT7B")
    image = Image.open("/data/pufanyi/project/Otter-2/pipeline/evaluation/test_data/test.jpg")
    response = model.generate("What is this?", image)
    print(response)
    response = model.generate("What is this?", image)
    print(response)
