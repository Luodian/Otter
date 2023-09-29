import base64
import io
import os
import shutil

import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from .base_eval_dataset import BaseEvalDataset

video_dir = "data_source/multi_hop_reasoning/"


class MultiHopBenchDataset(BaseEvalDataset):
    def __init__(self, dataset_path):
        super().__init__("MultiHopBenchDataset", dataset_path)
        cache_path = snapshot_download(repo_id=dataset_path, repo_type="dataset")
        self.df = load_dataset(os.path.join(cache_path, "multi-hop-reasoning.py"))
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        if not os.path.exists(os.path.join(video_dir, "images")):
            shutil.unpack_archive(os.path.join(cache_path, "images.zip"), video_dir)

    def evaluate(self, model, output_file=None):
        result = dict()
        for cur_data in tqdm(self.df["test"]):
            question_idx = cur_data["question_idx"]
            question = cur_data["question"]
            answer = cur_data["answer"]
            video_idx = cur_data["video_idx"]
            rationale = cur_data["rationale"]
            cur_data["video_root"] = os.path.join(video_dir, "images")
            response = model.generate(cur_data)
            results["question_idx"].append(question_idx)
            results["video_idx"].append(video_idx)
            results["question"].append(question)
            results["response"].append(response)

        df = pd.DataFrame(results)
        with pd.ExcelWriter(
            output_file,
            engine="xlsxwriter",
        ) as writer:
            df.to_excel(writer, index=False)

        print(f"MultiHopBenchDataset Evaluator: Result saved to {output_file}.")


if __name__ == "__main__":
    dataset = MHBenchDataset("ZhangYuanhan/multi-hop-reasoning")
    dataset.evaluate("123")
