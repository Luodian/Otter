import io
import torch
from typing import List
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image
from .base_model import BaseModel
from pipeline.train.train_utils import find_and_remove_tokens, get_image_attention_mask
import base64
import numpy as np


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


def get_single_formatted_prompt(question, image=None, answer="") -> List[str]:
    if answer == "":
        return [
            f"User:",
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


def get_formatted_prompt(questions, images, answers=""):
    single_prompt = False
    if not isinstance(questions, list):
        questions = [questions]
        single_prompt = True
    if not isinstance(images, list):
        images = [images]
    if not isinstance(answers, list):
        answers = [answers] * len(questions)
    result = []
    for question, image, answer in zip(questions, images, answers):
        result.append(get_single_formatted_prompt(question, image, answer))
    if single_prompt:
        return result[0]
    else:
        return result


class Idefics(BaseModel):
    def __init__(self, model_path: str = "HuggingFaceM4/idefics-9b-instruct", batch=8):
        super().__init__("idefics", model_path, max_batch_size=batch)
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
        self.patch_resize_transform = self.processor.image_processor.preprocess

    def generate(self, question, raw_image_data):
        formatted_prompt = get_formatted_prompt(question, raw_image_data)
        inputs = self.processor(formatted_prompt, return_tensors="pt").to(self.device)
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_new_tokens=768,
            temperature=0.2,
            do_sample=True,
            top_p=0.5,
        )
        generated_text = self.processor.batch_decode(generated_ids)
        results = list(map(lambda text: text.strip().split("Assistant:")[-1].split("<end_of_utterance>")[0].strip(), generated_text))
        if isinstance(question, str):
            return results[0]
        else:
            return results

    def prepare_labels(self, input_ids, eos_token_id, answer_token_id, endofchunk_token_id, fake_token_image_token_id, masking_number: int = -100):
        labels = torch.empty(input_ids.shape, dtype=torch.int64)
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
        forward_prompt = f"User:<fake_token_around_image><image><fake_token_around_image>{question}<end_of_utterance>\nAssistant:<answer>{answer}<end_of_utterance>"
        inputs = self.processor.tokenizer(forward_prompt, return_tensors="pt")
        vision_x = self.patch_resize_transform(image).unsqueeze(0).to(self.device)
        labels = self.prepare_labels(
            inputs["input_ids"],
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
                pixel_values=vision_x,
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                image_attention_mask=image_attention_mask.to(self.device),
                labels=labels.to(self.device),
                # input_ids=input_ids,
                # attention_mask=attention_mask,
                # image_attention_mask=image_attention_mask,
                # vision_x=vision_x,
                # labels=labels,
            ).loss
        return loss

    def eval_forward_batch(self, batch_questions, batch_options, batch_images):
        batch_size = len(batch_questions)
        all_option_losses = []
        tensor_images = [self.patch_resize_transform(image).unsqueeze(0) for image in batch_images]

        # Prepare batched inputs and put them on the device
        batch_input_ids = []
        batch_attention_mask = []
        batch_prompt = []

        for i in range(batch_size):
            question = batch_questions[i]
            option = batch_options[i]
            forward_prompt = f"User:<fake_token_around_image><image><fake_token_around_image>{question}<end_of_utterance>\nAssistant:<answer>{option}<end_of_utterance>"
            batch_prompt.append(forward_prompt)

        inputs = self.processor.tokenizer(batch_prompt, return_tensors="pt", padding="longest", truncation=True, max_length=512)
        batch_input_ids.append(inputs["input_ids"])
        batch_attention_mask.append(inputs["attention_mask"])

        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_attention_mask = torch.cat(batch_attention_mask, dim=0)
        batch_labels = self.prepare_labels(
            batch_input_ids,
            self.eos_token_id,
            self.answer_token_id,
            self.endofchunk_token_id,
            self.fake_token_image_token_id,
        )

        batch_input_ids, batch_labels, batch_attention_mask = find_and_remove_tokens(batch_input_ids, batch_labels, batch_attention_mask, self.answer_token_id, self.processor.tokenizer)

        # to device
        batch_image_tensors = torch.stack(tensor_images).to(self.device)
        batch_input_ids = batch_input_ids.to(self.device)
        batch_labels = batch_labels.to(self.device)
        batch_attention_mask = batch_attention_mask.to(self.device)

        # Perform batch inference
        with torch.no_grad():
            # Your forward function can go here, adjusted for batches
            outputs = self.model(
                pixel_values=batch_image_tensors.squeeze(2),
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                image_attention_mask=get_image_attention_mask(batch_input_ids, 1, self.processor.tokenizer).to(self.device),
                labels=batch_labels,
                # more arguments as needed
            )

        # Assuming `outputs.per_token_loss` contains the loss for each token for each item in the batch
        per_token_loss = outputs.per_token_loss  # Shape would be [batch_size, sequence_length]

        # Summing along the sequence length dimension to get per-item loss
        per_item_loss = torch.sum(per_token_loss, dim=1)  # Shape [batch_size]
        all_option_losses = np.split(per_item_loss, batch_size)

        return all_option_losses


if __name__ == "__main__":
    model = Idefics("/data/pufanyi/training_data/checkpoints/idefics-9b-instruct")
    print(
        model.generate(
            "What is in this image?",
            Image.open("/data/pufanyi/project/Otter-2/pipeline/evaluation/test_data/test.jpg"),
        )
    )
