import os
import re
import pandas as pd
from tqdm import tqdm, trange
from datasets import load_dataset
from .base_eval_dataset import BaseEvalDataset
import pytz
import datetime
import json

utc_plus_8 = pytz.timezone("Asia/Singapore")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)


class ScienceQADataset(BaseEvalDataset):
    def __init__(
        self, data_path: str = "Otter-AI/ScienceQA", *, split="test", cache_dir=None, default_output_path="./logs/ScienceQA", batch=1, debug=False, prompt='Please answer the question in the following format: "The answer is {A/B/C/D}".'
    ):
        super().__init__("ScienceQADataset", data_path, max_batch_size=batch)
        self.split = split
        self.data = load_dataset(data_path, split=self.split, cache_dir=cache_dir)
        self.default_output_path = default_output_path
        self.cur_datetime = utc_plus_8_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.debug = debug
        self.prompt = prompt

    def format_question(self, question, choices, answer):
        len_choices = len(choices)
        options = [chr(ord("A") + i) for i in range(len_choices)]
        answer = options[answer]
        choices_dict = dict(zip(options, choices))
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{self.prompt}\n{question}\n{choices_str}\n", choices_dict, answer

    def parse_pred_ans(self, pred_ans, options):
        match = re.search(r"The answer is ([A-D])", pred_ans)
        if match:
            return match.group(1)
        for c, option in options.items():
            option = option.strip()
            if option.upper() in pred_ans.upper():
                return c
        choices = set(options.keys())
        for selection in choices:
            if selection in pred_ans:
                return selection
        for selection in choices:
            if selection in pred_ans.upper():
                return selection
        return "other"

    def _evaluate(self, model, *, batch=1):
        if not os.path.exists(self.default_output_path):
            os.makedirs(self.default_output_path)

        output_file = os.path.join(self.default_output_path, f"{model.name}_scienceqa_eval_result_{self.cur_datetime}.json")
        result_file = os.path.join(self.default_output_path, f"{model.name}_scienceqa_eval_score_{self.cur_datetime}.json")
        results = []

        total = 0
        total_correct = 0

        for data in tqdm(self.data, desc="Evaluating", total=len(self.data)):
            question, choices_dict, answer = self.format_question(data["question"], data["choices"], data["answer"])
            output = model.generate(question, data["image"])
            phrased_output = self.parse_pred_ans(output, choices_dict)
            correct = phrased_output == answer
            if correct:
                total_correct += 1
            total += 1
            results.append(
                {
                    "question": data["question"],
                    "choices": data["choices"],
                    "answer": answer,
                    "output": output,
                    "prediction": phrased_output,
                    "correct": correct,
                }
            )
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

        score = total_correct / total
        print(f"ScienceQA Evaluator: Total: {total}")
        print(f"ScienceQA Evaluator: Total correct: {total_correct}")
        print(f"ScienceQA Evaluator: Score: {score}")
        with open(result_file, "w") as f:
            final_score = {
                "score": score,
                "total": total,
                "correct": total_correct,
            }
            json.dump(final_score, f, indent=4)

        print(f"ScienceQA Evaluator: Result saved to {os.path.abspath(output_file)}.")
        print(f"ScienceQA Evaluator: Score saved to {os.path.abspath(result_file)}.")


if __name__ == "__main__":
    dataset = ScienceQADataset(cache_dir="/data/pufanyi/cache")
    data = dataset.data
    print("=============================")
    import json

    print(json.dumps(data[1], indent=4))
    print("=============================")
    print(build_prompt(dataset.data[1], "QCM-ALE"))
    print("=============================")
