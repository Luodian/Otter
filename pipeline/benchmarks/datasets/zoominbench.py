import base64
import io
from PIL import Image
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os
import numpy as np
from datasets import load_dataset
from typing import Union
from .base_eval_dataset import BaseEvalDataset
from tqdm import tqdm
import datetime
import pytz
import re

utc_plus_8 = pytz.timezone("Asia/Singapore")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)


class ZoomInBenchDataset(BaseEvalDataset):
    def __init__(
        self,
        data_path: str = "Otter-AI/ZoomInBench",
        *,
        cache_dir: Union[str, None] = None,
        default_output_path: str = "./logs/ZoomInBench",
        split: str = "test",
        debug: bool = False,
        prompt='Please answer the question in the following format: "The answer is {A/B/C/D}."',
    ):
        super().__init__("ZoomInBench", data_path)

        self.default_output_path = default_output_path
        self.cur_datetime = utc_plus_8_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.data = load_dataset(data_path, split=split, cache_dir=cache_dir)
        self.debug = debug
        self.prompt = prompt

    def parse_pred_ans(self, pred_ans):
        match = re.search(r"The answer is ([A-D])", pred_ans)
        if match:
            return match.group(1)
        choices = ["A", "B", "C", "D"]
        for selection in choices:
            if selection in pred_ans:
                return selection
        for selection in choices:
            if selection in pred_ans.upper():
                return selection
        return "other"

    def _evaluate(self, model):
        model_score_dict = {}

        output_path = os.path.join(self.default_output_path, f"{model.name}_{self.cur_datetime}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        model_answer_path = os.path.join(output_path, f"{model.name}_answer.json")
        result_path = os.path.join(output_path, f"{model.name}_score.json")
        model_answer = {}

        score = 0
        num_data = 0

        for data in tqdm(self.data, desc="Evaluating", total=len(self.data)):
            question = f"{self.prompt} {data['instruction']}"
            if len(data["images"]) != 1:
                print(f"Warning: {data['id']} has {len(data['images'])} images.")
                print(f"Skipping {data['id']}")
                continue
            pred_ans = model.generate(question, data["images"][0])
            print(f"Question: {question}")
            print(f"Answer: {data['answer']}")
            print(f"Prediction: {pred_ans}")
            pred_ans = self.parse_pred_ans(pred_ans)
            print(f"Parsed prediction: {pred_ans}")
            num_data += 1
            if pred_ans == data["answer"]:
                score += 1
            model_answer[data["id"]] = {
                "question": question,
                "output": pred_ans,
                "answer": data["answer"],
            }
            with open(model_answer_path, "w") as f:
                json.dump(model_answer, f, indent=2)

        model_score_dict["score"] = score
        model_score_dict["total"] = len(self.data)
        model_score_dict["accuracy"] = score / len(self.data)

        with open(result_path, "w") as f:
            json.dump(model_score_dict, f, indent=2)

        print(f"Model answer saved to {model_answer_path}")
        print(f"Model score saved to {result_path}")
        print(json.dumps(model_score_dict, indent=2))

        return model_score_dict
