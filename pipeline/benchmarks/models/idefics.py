import io
import torch
from typing import List
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image
from .base_model import BaseModel
from pipeline.train.train_utils import find_and_remove_tokens, get_image_attention_mask
import base64


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


def get_formatted_prompt(question, image=None, answer="") -> List[str]:
    if answer == "":
        return [
            f"User",
            get_pil_image(image),
            question,
            "<end_of_utterance>\n",
            "Assistant:",
        ]
    else:
        return [
            f"User:",
            get_pil_image(image),
            question,
            "<end_of_utterance>\n",
            f"Assistant:<answer> {answer}",
            "<end_of_utterance>",
        ]


class Idefics(BaseModel):
    def __init__(self, model_path: str = "HuggingFaceM4/idefics-9b-instruct"):
        super().__init__("idefics", model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = IdeficsForVisionText2Text.from_pretrained(model_path, device_map={"": self.device}, torch_dtype=torch.bfloat16).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)

        if "<answer>" not in self.processor.tokenizer.special_tokens_map["additional_special_tokens"]:
            past_special_tokens = self.processor.tokenizer.special_tokens_map["additional_special_tokens"]
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>"] + past_special_tokens})

        self.fake_token_image_token_id = self.processor.tokenizer("<fake_token_around_image>", add_special_tokens=False)["input_ids"][-1]
        self.endofchunk_text = "<end_of_utterance>"
        self.endofchunk_token_id = self.processor.tokenizer(self.endofchunk_text, add_special_tokens=False)["input_ids"][-1]
        self.answer_token_id = self.processor.tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
        self.eos_token_id = self.processor.tokenizer(self.processor.tokenizer.eos_token, add_special_tokens=False)["input_ids"][-1]

    def generate(self, question: str, raw_image_data):
        formatted_prompt = get_formatted_prompt(question, raw_image_data)
        len_formatted_prompt = len(formatted_prompt[0]) + len(formatted_prompt[-1]) + 1
        inputs = self.processor(formatted_prompt, return_tensors="pt").to(self.device)
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_new_tokens=512,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text[0][len_formatted_prompt:].strip()

    def prepare_labels(self, input_ids, device_id, eos_token_id, answer_token_id, endofchunk_token_id, fake_token_image_token_id, masking_number: int = -100):
        labels = torch.empty(input_ids.shape, dtype=torch.int64).to(device_id, non_blocking=True)
        for i in range(input_ids.shape[0]):
            labels[i] = torch.where(input_ids[i] == eos_token_id, eos_token_id, masking_number)
            answer_token_ids_all = torch.where(input_ids[i] == answer_token_id)[0]
            endofchunk_token_ids_all = torch.where(input_ids[i] == endofchunk_token_id)[0]

            j = 0  # Counter for endofchunk_token_ids
            for answer_token_idx in answer_token_ids_all:
                # Find the closest endofchunk_token_id that is greater than answer_token_id
                while j < len(endofchunk_token_ids_all) and endofchunk_token_ids_all[j] < answer_token_idx:
                    j += 1

                if j < len(endofchunk_token_ids_all):
                    endofchunk_token_idx = endofchunk_token_ids_all[j]
                    labels[i, answer_token_idx + 1 : endofchunk_token_idx + 1] = input_ids[i, answer_token_idx + 1 : endofchunk_token_idx + 1]

                    # Increment j for the next iteration
                    j += 1

        labels[:, 0] = masking_number
        labels[labels == fake_token_image_token_id] = masking_number
        return labels

    def eval_forward(self, question, answer, image):
        formatted_prompt = get_formatted_prompt(question, image, answer)
        inputs = self.processor(formatted_prompt, add_end_of_utterance_token=False, return_tensors="pt").to(self.device)
        labels = self.prepare_labels(
            inputs["input_ids"],
            self.device,
            self.eos_token_id,
            self.answer_token_id,
            self.endofchunk_token_id,
            self.fake_token_image_token_id,
        )
        input_ids, labels, attention_mask = find_and_remove_tokens(
            inputs["input_ids"], labels, inputs["attention_mask"], self.answer_token_id, self.processor.tokenizer
        )  # find and remove certain tokens from input_ids, labels, and attention_mask
        # input_ids = inputs["input_ids"]
        # attention_mask = inputs["attention_mask"]
        image_attention_mask = get_image_attention_mask(input_ids, 1, self.processor.tokenizer)
        # vision_x = inputs["pixel_values"]
        # query = get_formatted_forward_prompt(question, answer)
        # tokens = self.tokenizer(query, return_tensors="pt")
        # input_ids = tokens["input_ids"]
        # attention_mask = tokens["attention_mask"]
        with torch.no_grad():
            loss = self.model(
                pixel_values=inputs["pixel_values"],
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_attention_mask=image_attention_mask,
                labels=labels,
                # input_ids=input_ids,
                # attention_mask=attention_mask,
                # image_attention_mask=image_attention_mask,
                # vision_x=vision_x,
                # labels=labels,
            ).loss
        return loss


if __name__ == "__main__":
    model = Idefics("/data/pufanyi/training_data/checkpoints/idefics-9b-instruct")
    print(
        model.generate(
            "What is in this image?",
            Image.open("/data/pufanyi/project/Otter-2/pipeline/evaluation/test_data/test.jpg"),
        )
    )
