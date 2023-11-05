import os
import datetime
from tqdm import tqdm, trange
from .base_eval_dataset import BaseEvalDataset
from datasets import load_dataset
import json
from typing import Union


class PopeDataset(BaseEvalDataset):
    def __init__(
        self,
        data_path="Otter-AI/POPE",
        split="test",
        default_output_path="./logs/POPE",
        cache_dir=None,
        batch_size=1,
    ):
        super().__init__("PopeDataset", data_path, max_batch_size=batch_size)
        print("Loading dataset from", data_path)
        self.data = load_dataset(data_path, split=split, cache_dir=cache_dir)
        print("Dataset loaded")
        self.default_output_path = default_output_path
        if not os.path.exists(default_output_path):
            os.makedirs(default_output_path)
        self.batch_gen_size = batch_size

    def parse_pred(self, text):
        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "").lower()
        words = text.split(" ")

        if "not" in words or "no" in words:
            return "no"
        else:
            return "yes"

    def _evaluate(self, model, batch=1):
        cur_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        output_path = os.path.join(self.default_output_path, f"pope_{model.name}_test_submit_{cur_datetime}.json")

        metrics = {
            "adversarial": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
            "popular": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
            "random": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
            "overall": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "yes_count": 0, "no_count": 0},
        }

        batch_size = self.batch_gen_size
        num_batches = len(self.data) // batch_size + 1

        with tqdm(total=len(self.data), desc="Evaluating") as pbar:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                batch_data = self.data[start_idx:end_idx]

                batch_questions = batch_data["question"]
                batch_answers = batch_data["answer"]
                batch_images = batch_data["image"]

                # if model has batch_generate, use it
                if hasattr(model, "batch_generate") and self.batch_gen_size > 1:
                    batch_responses = model.batch_generate(batch_questions, batch_images)
                else:
                    batch_responses = [model.generate(question, image) for question, image in zip(batch_questions, batch_images)]

                batch_preds = [self.parse_pred(response) for response in batch_responses]

                for j in range(len(batch_preds)):
                    answer = batch_answers[j]
                    pred = batch_preds[j]
                    category = batch_data["category"][j]

                    if answer == "yes":
                        metrics[category]["yes_count"] += 1
                        metrics["overall"]["yes_count"] += 1
                    else:
                        metrics[category]["no_count"] += 1
                        metrics["overall"]["no_count"] += 1

                    if pred == answer and pred == "yes":
                        metrics[category]["TP"] += 1
                        metrics["overall"]["TP"] += 1
                    elif pred == answer and pred == "no":
                        metrics[category]["TN"] += 1
                        metrics["overall"]["TN"] += 1
                    elif pred != answer and pred == "yes":
                        metrics[category]["FP"] += 1
                        metrics["overall"]["FP"] += 1
                    else:
                        metrics[category]["FN"] += 1
                        metrics["overall"]["FN"] += 1

                pbar.update(batch_size)

        for category in metrics:
            print(f"----------- {category} -----------")

            TP = metrics[category]["TP"]
            TN = metrics[category]["TN"]
            FP = metrics[category]["FP"]
            FN = metrics[category]["FN"]
            yes_count = metrics[category]["yes_count"]
            no_count = metrics[category]["no_count"]

            print("TP\tFP\tTN\tFN\t")
            print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

            if TP + FP == 0:
                metrics[category]["precision"] = precision = 0
            else:
                metrics[category]["precision"] = precision = float(TP) / float(TP + FP)

            if TP + FN == 0:
                metrics[category]["recall"] = recall = 0
            else:
                metrics[category]["recall"] = recall = float(TP) / float(TP + FN)

            if precision + recall == 0:
                metrics[category]["f1"] = f1 = 0
            else:
                metrics[category]["f1"] = f1 = 2 * precision * recall / float(precision + recall)

            metrics[category]["acc"] = acc = float(TP + TN) / float(TP + TN + FP + FN)

            if yes_count + no_count == 0:
                metrics[category]["yes_ratio"] = yes_ratio = 0
            else:
                metrics[category]["yes_ratio"] = yes_ratio = yes_count / float(yes_count + no_count)

            print("Accuracy: {}".format(acc))
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1 score: {}".format(f1))
            print("Yes ratio: {}".format(yes_ratio))

        print(f"----------- overall -----------")

        TP = metrics["overall"]["TP"]
        TN = metrics["overall"]["TN"]
        FP = metrics["overall"]["FP"]
        FN = metrics["overall"]["FN"]
        yes_count = metrics["overall"]["yes_count"]
        no_count = metrics["overall"]["no_count"]

        print("TP\tFP\tTN\tFN\t")
        print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

        metrics["overall"]["precision"] = precision = float(TP) / float(TP + FP)
        metrics["overall"]["recall"] = recall = float(TP) / float(TP + FN)
        metrics["overall"]["f1"] = f1 = 2 * precision * recall / float(precision + recall)
        metrics["overall"]["acc"] = acc = float(TP + TN) / float(TP + TN + FP + FN)
        metrics["overall"]["yes_ratio"] = yes_ratio = float(yes_count) / float(yes_count + no_count)

        print("Accuracy: {}".format(acc))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1 score: {}".format(f1))
        print("Yes ratio: {}".format(yes_ratio))

        output_f = open(output_path, "a")
        output_f.write(json.dumps(metrics) + "\n")
        output_f.close()
        return metrics
