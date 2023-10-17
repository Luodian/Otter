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


def get_single_formatted_prompt(instruction: str, images: List[Image.Image]) -> List[str]:
    end_of_utterance = "<end_of_utterance>\n"
    image_tag = "<image>"
    user_note = "User:"
    assistant_note = "Assistant:"
    change_role_tag = "<change_role>"
    instruction = instruction.replace(f"{image_tag}{user_note}", f"{user_note}{image_tag}")
    instruction = instruction.replace("GPT:", assistant_note)
    instruction = instruction.replace(user_note, f"{change_role_tag}{end_of_utterance}{change_role_tag}{user_note}")
    instruction = instruction.replace(assistant_note, f"{change_role_tag}{end_of_utterance}{change_role_tag}{assistant_note}")
    instruction = instruction.split(image_tag)
    assert len(instruction) == len(images) + 1, f"Number of images ({len(images)}) does not match number of image tags ({len(instruction) - 1})"
    instruction_with_image = []
    for i in range(len(images)):
        instruction_with_image.append(instruction[i].strip())
        instruction_with_image.append(images[i])
    instruction_with_image.append(instruction[-1].strip())

    final_instruction = []

    def format(text):
        if not isinstance(text, str):
            return [text]
        text = text.strip()
        text = text.split(change_role_tag)
        return text

    for ins in instruction_with_image:
        for formatted_ins in format(ins):
            if formatted_ins is None or formatted_ins == "":
                continue
            if final_instruction == [] and formatted_ins == end_of_utterance:
                continue
            if isinstance(formatted_ins, str):
                final_instruction.append(formatted_ins.strip())
            else:
                final_instruction.append(formatted_ins)

    if final_instruction[-1] == "Assistant:<answer>":
        final_instruction[-1] = "Assistant:"
    elif final_instruction[-1] != end_of_utterance:
        final_instruction.append(end_of_utterance)

    return final_instruction

    # if answer == "":
    #     return [
    #         f"User:",
    #         image,
    #         question,
    #         "<end_of_utterance>\n",
    #         "Assistant:",
    #     ]
    # else:
    #     return [
    #         f"User:",
    #         image,
    #         question,
    #         "<end_of_utterance>\n",
    #         f"Assistant:<answer> {answer}",
    #         "<end_of_utterance>",
    #     ]


def get_formatted_prompt(questions, images):
    instructions = []
    for ques, img in zip(questions, images):
        instructions.append(get_single_formatted_prompt(ques, img))
    return instructions


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

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        instructions = get_formatted_prompt(batch_text, batch_images)
        inputs = self.processor(instructions, return_tensors="pt").to(self.device)
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        with torch.inference_mode():
            with self.autocast():
                generated_ids = unwrap_model(self.model).generate(
                    **inputs,
                    eos_token_id=exit_condition,
                    bad_words_ids=bad_words_ids,
                    min_new_tokens=min_generation_length,
                    max_new_tokens=max_generation_length,
                    # num_beams=num_beams,
                    # length_penalty=length_penalty,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.5,
                )
        generated_text = self.processor.batch_decode(generated_ids)
        results = list(map(lambda text: text.split("Assistant:")[-1].split(self.endofchunk_text)[0].strip(), generated_text))
        # print(max_generation_length)
        # print(results)
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
        return f"<image>User: {question} Please answer in short words. GPT:<answer>{answer if answer is not None else ''}{self.endofchunk_text if answer is not None else ''}"

    def get_caption_prompt(self, caption=None) -> str:
        return f"<image>User: What does the image describe? GPT:<answer>{caption if caption is not None else ''}{self.endofchunk_text if caption is not None else ''}"


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


if __name__ == "__main__":
    s = "<image>User: what is the brand on the bottle? Please answer in short words. GPT:<answer>"
    print(get_single_formatted_prompt(s, ["An image"]))
