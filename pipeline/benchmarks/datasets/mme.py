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

utc_plus_8 = pytz.timezone("Asia/Singapore")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)

eval_type_dict = {
    "Perception": [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "ocr",
    ],
    "Cognition": ["commonsense", "numerical", "text", "code"],
}


class MMEDataset(BaseEvalDataset):
    def decode_base64_to_image(self, base64_string):
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image

    def __init__(
        self,
        data_path: str = "Otter-AI/MME",
        *,
        cache_dir: Union[str, None] = None,
        default_output_path: str = "./logs/MME",
        split: str = "test",
        debug: bool = False,
    ):
        super().__init__("MMEDataset", data_path)

        self.default_output_path = default_output_path
        self.cur_datetime = utc_plus_8_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.data = load_dataset(data_path, split=split, cache_dir=cache_dir)
        self.debug = debug

        self.category_data = {}
        # for idx in range(len(self.ids)):
        for item in tqdm(self.data, desc="Loading data"):
            id = item["id"]
            category = id.split("_")[0].lower()
            question = item["instruction"]
            answer = item["answer"]
            image_id = item["image_ids"][0]
            image = item["images"][0]

            data = {"question": question, "answer": answer, "image": image}

            if category in eval_type_dict["Cognition"]:
                eval_type = "Cognition"
            elif category in eval_type_dict["Perception"]:
                eval_type = "Perception"
            else:
                raise ValueError(f"Unknown category {category}")

            if eval_type not in self.category_data:
                self.category_data[eval_type] = {}

            if category not in self.category_data[eval_type]:
                self.category_data[eval_type][category] = {}

            if image_id not in self.category_data[eval_type][category]:
                self.category_data[eval_type][category][image_id] = []

            self.category_data[eval_type][category][image_id].append(data)

    def parse_pred_ans(self, pred_ans):
        pred_ans = pred_ans.lower().strip().replace(".", "")
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]
            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"
        return pred_label

    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }

        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds)

        clean_gts = []
        clean_preds = []
        other_num = 0
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0])
        precision = precision_score(clean_gts, clean_preds, average="binary")
        recall = recall_score(clean_gts, clean_preds, average="binary")
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        for key, value in metric_dict.items():
            if isinstance(value, np.int64):
                metric_dict[key] = int(value)

        return metric_dict

    def _evaluate(self, model):
        model_score_dict = {}

        self.default_output_path = os.path.join(self.default_output_path, f"{model.name}_{self.cur_datetime}")
        if not os.path.exists(self.default_output_path):
            os.makedirs(self.default_output_path)

        for eval_type in self.category_data.keys():
            print("===========", eval_type, "===========")

            scores = 0
            task_score_dict = {}
            for task_name in tqdm(self.category_data[eval_type].keys(), desc=f"Evaluating {eval_type}"):
                img_num = len(self.category_data[eval_type][task_name])
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []
                for image_pair in tqdm(self.category_data[eval_type][task_name].values(), desc=f"Evaluating {eval_type} {task_name}"):
                    assert len(image_pair) == 2
                    img_correct_num = 0

                    for item in image_pair:
                        question = item["question"]
                        image = item["image"]
                        gt_ans = item["answer"].lower().strip().replace(".", "")
                        response = model.generate(question, image)
                        if self.debug:
                            print(f"\n# Query: {question}")
                            print(f"\n# Response: {response}")
                        pred_ans = self.parse_pred_ans(response)

                        assert gt_ans in ["yes", "no"]
                        assert pred_ans in ["yes", "no", "other"]

                        gts.append(gt_ans)
                        preds.append(pred_ans)

                        if gt_ans == pred_ans:
                            img_correct_num += 1

                        if pred_ans not in ["yes", "no"]:
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict["acc_plus"] = acc_plus

                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v * 100

                task_score_dict[task_name] = task_score
                scores += task_score

                output_path = os.path.join(self.default_output_path, f"{task_name}.json")
                with open(output_path, "w") as f:
                    json.dump(metric_dict, f)

            print(f"total score: {scores}")
            for task_name, score in task_score_dict.items():
                print(f"\t {task_name} score: {score}")
