from transformers import FuyuForCausalLM, AutoTokenizer, FuyuProcessor, FuyuImageProcessor
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


class Fuyu(BaseModel):
    def __init__(self, model_path: str = "adept/fuyu-8b"):
        super().__init__("fuyu", model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = FuyuForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(image_processor=self.image_processor, tokenizer=self.tokenizer)

    def generate(self, text_prompt: str, raw_image_data: str):
        raw_image_data = get_pil_image(raw_image_data)
        # make sure the image is in RGB format and resize to match the width
        max_height, max_width = 1080, 1920
        raw_image_data = raw_image_data.convert("RGB")
        raw_image_data.thumbnail((max_width, max_height), Image.ANTIALIAS)
        model_inputs = self.processor(text=text_prompt, images=[raw_image_data], device=self.device)
        for k, v in model_inputs.items():
            model_inputs[k] = v.to(self.device)

        generation_output = self.model.generate(**model_inputs, max_new_tokens=1024, pad_token_id=self.tokenizer.eos_token_id)
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
        return generation_text[0].split("\x04")[1].strip(" ").strip("\n")

    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for Fuyu
        pass

    def batch_generate(self, batch_text_prompts: list[str], batch_raw_image_data: list[Image.Image]):
        batch_raw_image_data = [image.convert("RGB") for image in batch_raw_image_data]
        # Prepare the inputs for the model
        model_inputs = self.processor(text=batch_text_prompts, images=batch_raw_image_data, device=self.device)

        for k, v in model_inputs.items():
            model_inputs[k] = v.to(self.device)

        # Forward pass to generate the batch output
        generation_output = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.eos_token_id
            # Add any other parameters similar to Idefics.generate
        )

        # Decode the generated text
        generation_texts = self.processor.batch_decode(generation_output, skip_special_tokens=True)

        # Process the output
        processed_output = [text.split("\x04")[1].strip() for text in generation_texts]

        return processed_output


if __name__ == "__main__":
    model = Fuyu()
    print(model.generate("Generate a coco-style caption.\n", Image.open("/home/luodian/projects/Otter/archived/test_images/rabbit.png").convert("RGB")))
