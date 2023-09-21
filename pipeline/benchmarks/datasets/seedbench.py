import os
import json
from PIL import Image
import numpy as np
import torch
from otter_ai import OtterForConditionalGeneration
import transformers
from tqdm import tqdm
from .base_eval_dataset import BaseEvalDataset
from datasets import load_dataset


Image.MAX_IMAGE_PIXELS = 100_000_000


class SEEDBenchDataset(BaseEvalDataset):
    def __init__(self, data_path="Otter-AI/SEEDBench", *, split="train", cache_dir=None):
        super().__init__("SEEDBenchDataset", data_path)
        self.data = load_dataset("Otter-AI/SEEDBench", split=split, cache_dir=cache_dir)

    def evaluate(self, model):
        num_correct = 0
        for data_dict in tqdm(self.data, total=len(self.data), desc="Evaluating"):
            image = data_dict["image"]
            question = data_dict["question"]
            answer = data_dict["answer"]
            options = [
                data_dict["choice_a"],
                data_dict["choice_b"],
                data_dict["choice_c"],
                data_dict["choice_d"],
            ]

            option_losses = []
            for option in options:
                option_losses.append(model.eval_forward(question, option, image).items())

            prediction_idx = np.argmin(option_losses)
            prediction = ["A", "B", "C", "D"][prediction_idx]
            if prediction == answer:
                num_correct += 1

        accuracy = num_correct / len(self.data) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy


if __name__ == "__main__":
    dataset = SEEDBenchDataset("/data/joshua/datasets/SEEDBench/SEED-Bench.json", "/data/joshua/datasets/SEEDBench/SEED-Bench-image")
    for data in dataset:
        print(data)
        break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # checkpoint = "/data/pufanyi/training_data/checkpoints/idefics-9b-instruct"
    checkpoint = "/data/pufanyi/training_data/checkpoints/OTTER-Image-MPT7B"
    model = OtterForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    dataset.evaluate(model, tokenizer)
