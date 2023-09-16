import io
import torch
from typing import List
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image
from .base_model import BaseModel


def get_pil_image(raw_image_data) -> Image.Image:
    return Image.open(io.BytesIO(raw_image_data["bytes"]))


def get_formatted_prompt(prompt: str, image) -> List[str]:
    return [
        f"User: {prompt}",
        get_pil_image(image),
        "<end_of_utterance>",
        "\nAssistant:",
    ]


class Idefics(BaseModel):
    def __init__(self, model_path: str = "HuggingFaceM4/idefics-9b-instruct"):
        super().__init__("idefics", model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = model_path
        self.model = IdeficsForVisionText2Text.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16).to(self.device)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def generate(self, question: str, raw_image_data: Image.Image):
        formatted_prompt = get_formatted_prompt(question, raw_image_data)
        len_formatted_prompt = len(formatted_prompt[0]) + len(formatted_prompt[-1]) + 1
        inputs = self.processor(formatted_prompt, return_tensors="pt").to(self.device)
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        generated_ids = self.model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=500)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text[0][len_formatted_prompt:].strip()


if __name__ == "__main__":
    model = Idefics("/data/pufanyi/training_data/checkpoints/idefics-9b-instruct")
    print(model.generate("What is in this image?", Image.open("/data/pufanyi/project/Otter-2/pipeline/evaluation/test_data/test.jpg")))
