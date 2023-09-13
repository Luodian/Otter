import base64
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from .base_evel_dataset import BaseEvalDataset


class MHBenchDataset(BaseEvalDataset):
    def __init__(self, dataset_path):
        self.df = load_dataset(dataset_path)

    def evaluate(self, model, output_file=None):
        for cur_data in tqdm(self.df["test"]):
            question_idx = cur_data["question_idx"]
            question = cur_data["question"]
            answer = cur_data["answer"]
            video_idx = cur_data["video_idx"]
            rationale = cur_data["rationale"]
            response = model.generate(cur_data)


if __name__ == "__main__":
    dataset = MHBenchDataset("ZhangYuanhan/multi-hop-reasoning")
    dataset.evaluate("123")
