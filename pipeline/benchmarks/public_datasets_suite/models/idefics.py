from pipeline.benchmarks.public_datasets_suite.eval_model import BaseEvalModel
import io
import torch
from typing import List
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image
from pipeline.train.train_utils import find_and_remove_tokens, get_image_attention_mask
from pipeline.benchmarks.public_datasets_suite.models.utils import unwrap_model
import base64
import numpy as np
from contextlib import suppress
import re
import json


class EvalModel(BaseEvalModel):
    def __init__(self, model_args):
        if "model_path" in model_args:
            model_path = model_args["model_path"]
        else:
            model_path = "HuggingFaceM4/idefics-9b-instruct"
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

        # autocast
        self.autocast = get_autocast(model_args["precision"])
        self.cast_dtype = get_cast_dtype(model_args["precision"])

        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        self.token_around_image = "<fake_token_around_image>"
        self.image_token = "<image>"

    def get_list_image_vision_x(self, images: List[Image.Image]) -> torch.Tensor:
        return self.image_processor.preprocess(images, return_tensors="pt").to(self.device)

    def get_vision_x(self, batch_images: List[List[Image.Image]]) -> torch.Tensor:
        vision_x = [self.get_list_image_vision_x(images) for images in batch_images]
        return torch.stack(vision_x).to(self.device)

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        # print(json.dumps(batch_text, indent=4))
        # instructions = get_formatted_prompt(batch_text, batch_images)
        # inputs = self.processor(batch_text, return_tensors="pt").to(self.device)
        lang_x = self.tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
        )
        vision_x = self.get_vision_x(batch_images)
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        image_attention_mask = get_image_attention_mask(lang_x["input_ids"], vision_x.shape[1], self.tokenizer)
        # print(vision_x.shape, lang_x["input_ids"].shape, lang_x["attention_mask"].shape, image_attention_mask.shape)
        generated_ids = unwrap_model(self.model).generate(
            pixel_values=vision_x.to(self.model.device),
            input_ids=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            image_attention_mask=image_attention_mask.to(self.model.device),
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_new_tokens=max_generation_length,
            min_new_tokens=min_generation_length,
        )
        # generated_ids = unwrap_model(self.model).generate(
        #     **inputs,
        #     eos_token_id=exit_condition,
        #     bad_words_ids=bad_words_ids,
        #     min_new_tokens=min_generation_length,
        #     max_new_tokens=max_generation_length,
        #     # num_beams=num_beams,
        #     # length_penalty=length_penalty,
        #     temperature=0.2,
        #     do_sample=True,
        #     top_p=0.5,
        # )
        generated_text = self.processor.batch_decode(generated_ids)
        results = list(map(lambda text: text.split("Assistant:")[-1].split(self.endofchunk_text)[0].strip(), generated_text))
        # print(max_generation_length)
        # print(json.dumps(results, indent=4))
        return results

    def get_logits(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
        clear_conditioned_layers: bool = False,
    ):
        with torch.inference_mode():
            with self.autocast():
                outputs = self.model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    clear_conditioned_layers=clear_conditioned_layers,
                    past_key_values=past_key_values,
                    use_cache=(past_key_values is not None),
                )
        return outputs

    def get_vqa_prompt(self, question, answer=None) -> str:
        # return f"Image:{self.token_around_image}{self.image_token}{self.token_around_image}Question: {question} Answer: {answer if answer is not None else ''}\n{self.endofchunk_text}"
        return f"{self.token_around_image}{self.image_token}{self.token_around_image}User: {question} Please answer in short words.<end_of_utterance>\nAssistant:{f'{answer}{self.endofchunk_text}' if answer is not None else ''}"  # 14.36
        # return f"User: <fake_token_around_image><image><fake_token_around_image>{question}<end_of_utterance>\nAssistant:{f'{answer}{self.endofchunk_text}' if answer is not None else ''}" # 5.94

    def get_caption_prompt(self, caption=None) -> str:
        return f"{self.token_around_image}{self.image_token}{self.token_around_image}User: What does the image describe?<end_of_utterance>\nAssistant:{f'{caption}{self.endofchunk_text}' if caption is not None else ''}"


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


# if __name__ == "__main__":
#     s = "<image>User: what is the brand on the bottle? Please answer in short words. GPT:<answer>"
#     print(get_single_formatted_prompt(s, ["An image"]))
