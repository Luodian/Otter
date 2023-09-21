import base64
import io
import os
import sys

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

sys.path.append("/mnt/petrelfs/zhangyuanhan/Otter/pipeline/evaluation/eval_datasets/")
import shutil

# from .base_evel_dataset import BaseEvalDataset
from base_eval_dataset import BaseEvalDataset
from huggingface_hub import snapshot_download

current_file_path = os.path.abspath(__file__)
upper_folder_path = os.path.dirname(os.path.dirname(current_file_path))

video_dir = os.path.join(upper_folder_path, "data_source/multi_hop_reasoning/")


class MultiHopBenchDataset(BaseEvalDataset):
    def __init__(self, dataset_path, task="Generation", output_root=None):
        super().__init__("MultiHopBenchDataset", dataset_path)
        self.task = task
        self.output_root = output_root
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        cache_path = snapshot_download(repo_id=dataset_path, repo_type="dataset")
        self.df = load_dataset(os.path.join(cache_path, "multi-hop-reasoning.py"), task)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        if not os.path.exists(os.path.join(video_dir, "videos")):
            shutil.unpack_archive(os.path.join(cache_path, "videos.zip"), video_dir)
        self.video_dir = os.path.join(video_dir, "videos")

    def evaluate(self, model):
        results = dict()
        results["question_idx"] = []
        results["video_idx"] = []
        results["question"] = []
        results["response"] = []
        output_path = f"{self.output_root}/{model.name}.xlsx"
        for cur_data in tqdm(self.df["test"]):
            question_idx = cur_data["question_idx"]
            question = cur_data["question"]
            video_idx = cur_data["video_idx"]
            rationale = cur_data["rationale"]

            video_path = os.path.join(self.video_dir, cur_data["video_idx"] + ".mp4")
            if os.path.exists(video_path):
                cur_data["video_path"] = video_path
            elif os.path.exists(video_path.replace("mp4", "MP4")):
                video_path = video_path.replace("mp4", "MP4")
                cur_data["video_path"] = video_path
            else:
                sys.exit(f"video path:{video_path} does not exist, please check")

            response = model.generate(cur_data)
            results["question_idx"].append(question_idx)
            results["video_idx"].append(video_idx)
            results["question"].append(question)
            results["response"].append(response)

        df = pd.DataFrame(results)
        with pd.ExcelWriter(
            output_path,
            engine="xlsxwriter",
        ) as writer:
            df.to_excel(writer, index=False)

        print(f"MultiHopBenchDataset Evaluator: Result saved to {output_path}.")


if __name__ == "__main__":
    dataset = MultiHopBenchDataset("ZhangYuanhan/multi-hop-reasoning", task="Generation", output_root="/mnt/petrelfs/zhangyuanhan/Otter/pipeline/evaluation/evaluation_result")
    video_dir = os.path.join(video_dir, "videos")
    for cur_data in tqdm(dataset.df["test"]):
        cur_path = f"{video_dir}/{cur_data['video_idx']}.mp4"
        if not os.path.exists(cur_path):
            cur_path = cur_path.replace("mp4", "MP4")
            if not os.path.exists(cur_path):
                print("not exists", cur_path)

        if cur_data["object_description"] != "None" and "None" in cur_data["object_description"]:
            print(cur_path)
            import pdb

            pdb.set_trace()