from PIL import Image
import io
import torch
from transformers import AutoProcessor
from transformers import IdeficsForVisionText2Text
from .base_model import BaseModel
from pipeline.train.train_utils import get_image_attention_mask


def get_pil_image(raw_image_data) -> Image.Image:
    return Image.open(io.BytesIO(raw_image_data["bytes"]))


def get_formatted_prompt(question: str, no_image_flag: str) -> str:
    if no_image_flag:
        return f"User:{question}<end_of_utterance>\nAssistant:<answer>"
    else:
        return f"User:<fake_token_around_image><image><fake_token_around_image>{question}<end_of_utterance>\nAssistant:<answer>\n"


def get_formatted_forward_prompt(question: str, answer: str, no_image_flag: bool) -> str:
    if no_image_flag:
        return f"User:{question}<end_of_utterance>\nAssistant:<answer>\n{answer}<end_of_utterance>"
    else:
        return f"User:<fake_token_around_image><image><fake_token_around_image>{question}<end_of_utterance>\nAssistant:<answer>\n{answer}<end_of_utterance>"


class IdeficsOtter(BaseModel):
    def __init__(
        self,
        model_path: str,
        processor: str = "HuggingfaceM4/idefics-9b-instruct",
    ) -> None:
        super().__init__("idefics_otter", model_path)
        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        self.model = IdeficsForVisionText2Text.from_pretrained(model_path, **kwargs)
        self.processor = AutoProcessor.from_pretrained(processor)
        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.model.eval()

    def get_vision_x(self, input_data):
        if isinstance(input_data, Image.Image):
            if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
                vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(self.model.parameters()).dtype)
            else:
                vision_x = self.image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image.")
        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)
        return vision_x

    def generate(self, question, raw_image_data, no_image_flag=False):
        input_data = get_pil_image(raw_image_data)
        vision_x = self.get_vision_x(input_data)
        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)
        # vision_x = self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(0)
        lang_x = self.tokenizer(
            [
                get_formatted_prompt(question, no_image_flag=no_image_flag),
            ],
            return_tensors="pt",
        )
        image_attention_mask = get_image_attention_mask(lang_x["input_ids"], 1, self.tokenizer)
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        generated_text = self.model.generate(
            pixel_values=vision_x.to(self.model.device),
            input_ids=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            image_attention_mask=image_attention_mask.to(self.model.device),
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_new_tokens=512,
        )
        output = self.tokenizer.decode(generated_text[0])

        return output

    def eval_forward(self, question, answer, image):
        query = get_formatted_forward_prompt(question, answer)
        lang_x = self.tokenizer([query], return_tensors="pt")
        input_ids = lang_x["input_ids"]
        attention_mask = lang_x["attention_mask"]
        image_attention_mask = get_image_attention_mask(input_ids, 1, self.tokenizer)
        vision_x = self.get_vision_x(image)
        # query = get_formatted_forward_prompt(question, answer)
        # tokens = self.tokenizer(query, return_tensors="pt")
        # input_ids = tokens["input_ids"]
        # attention_mask = tokens["attention_mask"]
        with torch.no_grad():
            vision_x = self.get_vision_x(image)
            loss = self.model(
                pixel_values=vision_x.to(self.model.device), 
                lang_x=input_ids.to(self.model.device), 
                attention_mask=attention_mask.to(self.model.device),
                image_attention_mask=image_attention_mask.to(self.model.device)
            )[0]
        return loss


if __name__ == "__main__":
    model = IdeficsOtter("/data/pufanyi/training_data/checkpoints/otter_idefics9b_0830")
    image = Image.open("/data/pufanyi/project/Otter-2/pipeline/evaluation/test_data/test.jpg")
    response = model.generate("What is this?", image)
    print(response)
    response = model.generate("What is this?", image)
    print(response)
