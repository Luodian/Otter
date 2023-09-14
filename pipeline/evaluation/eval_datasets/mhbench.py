import base64
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import os

from .base_evel_dataset import BaseEvalDataset

from huggingface_hub import snapshot_download

import shutil

video_dir = "data_source/multi_hop_reasoning/"


class MHBenchDataset(BaseEvalDataset):
    def __init__(self, dataset_path):
        cache_path = snapshot_download(repo_id=dataset_path, repo_type="dataset")
        self.df = load_dataset(os.path.join(cache_path, "multi-hop-reasoning.py"))
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        if not os.path.exists(os.path.join(video_dir, "images")):
            shutil.unpack_archive(os.path.join(cache_path, "images.zip"), video_dir)

    def evaluate(self, model, output_file=None):
        for cur_data in tqdm(self.df["test"]):
            question_idx = cur_data["question_idx"]
            question = cur_data["question"]
            answer = cur_data["answer"]
            video_idx = cur_data["video_idx"]
            rationale = cur_data["rationale"]
            cur_data["video_root"] = os.path.join(video_dir, "images")
            response = model.generate(cur_data)


if __name__ == "__main__":
    dataset = MHBenchDataset("ZhangYuanhan/multi-hop-reasoning")
    dataset.evaluate("123")
