import base64
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset


class MHBenchDataset(object):
    def __init__(self, data_file):
        self.df = load_dataset(data_file)

    def evaluate(self, model, output_file=None):
        for cur_data in tqdm(self.df["test"]):
            question_idx = cur_data["question_idx"]
            question = cur_data["question"]
            answer = cur_data["answer"]
            video_idx = cur_data["video_idx"]
            rationale = cur_data["rationale"]
            # import pdb;pdb.set_trace()
            response = model.generate(cur_data)
            import pdb;pdb.set_trace()


if __name__ == "__main__":
    dataset = MHBenchDataset("ZhangYuanhan/multi-hop-reasoning")
    dataset.evaluate("123")

