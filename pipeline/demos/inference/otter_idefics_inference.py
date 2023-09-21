import mimetypes
import sys
import time
from typing import Union

import numpy
import requests
import yaml
from PIL import Image
from transformers import AutoProcessor, IdeficsForVisionText2Text

sys.path.append("../..")
import contextlib

from pipeline.demos.inference.utils import get_image, print_colored
from pipeline.train.train_utils import get_image_attention_mask
from pipeline.utils.general import DualOutput

requests.packages.urllib3.disable_warnings()
import torch


class TestOtterIdefics:
    def __init__(self, checkpoint) -> None:
        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        self.model = IdeficsForVisionText2Text.from_pretrained(checkpoint, **kwargs)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.model.eval()

    def generate(self, image, prompt, no_image_flag=False):
        input_data = image
        if isinstance(input_data, Image.Image):
            if no_image_flag:
                vision_x = torch.zeros(1, 1, 3, 224, 224, dtype=next(self.model.parameters()).dtype)
            else:
                vision_x = self.image_processor.preprocess([input_data], return_tensors="pt").unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image.")

        lang_x = self.tokenizer(
            [
                self.get_formatted_prompt(prompt, no_image_flag=no_image_flag),
            ],
            return_tensors="pt",
        )

        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)
        # vision_x = self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(0)
        lang_x = self.tokenizer(
            [
                self.get_formatted_prompt(prompt, no_image_flag=no_image_flag),
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
            num_beams=3,
            max_new_tokens=512,
        )
        output = self.tokenizer.decode(generated_text[0])
        output = output.split("<answer>")[1].lstrip("\n").strip().replace("<end_of_utterance>", "")
        return output

    def get_formatted_prompt(self, question: str, no_image_flag: str) -> str:
        if no_image_flag:
            return f"User:{question}<end_of_utterance>\nAssistant:<answer>"
        else:
            return f"User:<fake_token_around_image><image><fake_token_around_image>{question}<end_of_utterance>\nAssistant:<answer>\n"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="The path to the checkpoint.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    checkpoint = args.checkpoint
    # checkpoint = "/home/luodian/projects/checkpoints/Otter-Idefics-IT50-TEXT1K"
    model = TestOtterIdefics(checkpoint=args.checkpoint)

    while True:
        yaml_file = input("Enter the path to the yaml file: (or 'q' to quit): ")
        if yaml_file == "q":
            break
        with open(yaml_file, "r") as file:
            test_data_list = yaml.safe_load(file)

        log_file_path = yaml_file.replace(".yaml", "_results_log.txt")
        with open(log_file_path, "w") as log_file:
            for test_data in test_data_list:
                image_path = test_data.get("image_path", "")
                question = test_data.get("question", "")

                image = get_image(image_path)
                no_image_flag = not bool(image_path)

                response = model.generate(prompt=question, image=image, no_image_flag=no_image_flag)

                # Print results to console
                print(f"image_path: {image_path}")
                print_colored(f"question: {question}", color_code="\033[92m")
                print_colored(f"answer: {response}", color_code="\033[94m")
                print("-" * 150)

                log_file.write(f"image_path: {image_path}\n")
                log_file.write(f"question: {question}\n")
                log_file.write(f"answer: {response}\n\n")


if __name__ == "__main__":
    main()
