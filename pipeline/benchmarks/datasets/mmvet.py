import base64
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from .base_eval_dataset import BaseEvalDataset
from collections import Counter
from typing import Union
import numpy as np
from openai import OpenAI
import time
import json
import pytz
import datetime
from Levenshtein import distance

utc_plus_8 = pytz.timezone("Asia/Singapore")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)

MM_VET_PROMPT = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
"""


class MMVetDataset(BaseEvalDataset):
    def __init__(
        self,
        data_path: str = "Otter-AI/MMVet",
        gpt_model: str = "gpt-4-0613",
        *,
        api_key: str,
        split: str = "test",
        cache_dir: Union[str, None] = None,
        default_output_path: str = "./logs/MMVet",
        num_run: int = 1,
        prompt: str = MM_VET_PROMPT,
        decimail_places: int = 1,  # number of decimal places to round to
        debug: bool = False,
    ):
        super().__init__("MMVetDataset", data_path)
        self.df = load_dataset(data_path, split=split, cache_dir=cache_dir).to_pandas()
        self.default_output_path = default_output_path
        self.prompt = prompt
        self.gpt_model = gpt_model
        self.num_run = num_run
        self.decimal_places = decimail_places
        self.api_key = api_key
        self.cur_datetime = utc_plus_8_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.debug = debug
        self.prepare()
        self.client = OpenAI(api_key=api_key)

    def prepare(self):
        self.counter = Counter()
        self.cap_set_list = []
        self.cap_set_counter = []
        self.len_data = 0
        self.caps = {}

        for index, row in self.df.iterrows():
            self.caps[row["id"]] = row["capability"]

        for cap in self.df["capability"]:
            cap = set(cap)
            self.counter.update(cap)
            if cap not in self.cap_set_list:
                self.cap_set_list.append(cap)
                self.cap_set_counter.append(1)
            else:
                self.cap_set_counter[self.cap_set_list.index(cap)] += 1

            self.len_data += 1

        sorted_list = self.counter.most_common()
        self.columns = [k for k, v in sorted_list]
        self.columns.append("total")
        self.columns.append("std")
        self.columns.append("runs")
        self.result1 = pd.DataFrame(columns=self.columns)

        cap_set_sorted_indices = np.argsort(-np.array(self.cap_set_counter))
        new_cap_set_list = []
        new_cap_set_counter = []
        for index in cap_set_sorted_indices:
            new_cap_set_list.append(self.cap_set_list[index])
            new_cap_set_counter.append(self.cap_set_counter[index])

        self.cap_set_list = new_cap_set_list
        self.cap_set_counter = new_cap_set_counter
        self.cap_set_names = ["_".join(list(cap_set)) for cap_set in self.cap_set_list]

        self.columns2 = self.cap_set_names
        self.columns2.append("total")
        self.columns2.append("std")
        self.columns2.append("runs")
        self.result2 = pd.DataFrame(columns=self.columns2)

    def get_output_file_name(self, model, *, output_path: str = None, num_run: int = 1) -> str:
        if output_path is None:
            result_path = self.default_output_path
        else:
            result_path = output_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        model_results_file = os.path.join(result_path, f"{model.name}.json")
        grade_file = f"{model.name}-{self.gpt_model}-grade-{num_run}runs-{self.cur_datetime}.json"
        grade_file = os.path.join(result_path, grade_file)
        cap_score_file = f"{model.name}-{self.gpt_model}-cap-score-{num_run}runs-{self.cur_datetime}.csv"
        cap_score_file = os.path.join(result_path, cap_score_file)
        cap_int_score_file = f"{model.name}-{self.gpt_model}-cap-int-score-{num_run}runs-{self.cur_datetime}.csv"
        cap_int_score_file = os.path.join(result_path, cap_int_score_file)
        return model_results_file, grade_file, cap_score_file, cap_int_score_file

    def _evaluate(self, model):
        model_results_file, grade_file, cap_score_file, cap_int_score_file = self.get_output_file_name(model)

        if os.path.exists(grade_file):
            with open(grade_file, "r") as f:
                grade_results = json.load(f)
        else:
            grade_results = {}

        def need_more_runs():
            need_more_runs = False
            if len(grade_results) > 0:
                for k, v in grade_results.items():
                    if len(v["score"]) < self.num_run:
                        need_more_runs = True
                        break
            return need_more_runs or len(grade_results) < self.len_data

        print(f"grade results saved to {grade_file}")
        while need_more_runs():
            for j in range(self.num_run):
                print(f"eval run {j}")
                for _, line in tqdm(self.df.iterrows(), total=len(self.df)):
                    id = line["id"]
                    # if sub_set is not None and id not in sub_set:
                    #     continue
                    if id in grade_results and len(grade_results[id]["score"]) >= (j + 1):
                        continue

                    model_pred = model.generate(line["instruction"], line["images"][0])
                    if self.debug:
                        print(f"# Query: {line['instruction']}")
                        print(f"# Response: {model_pred}")
                        print(f"# Ground Truth: {line['answer']}")

                    question = (
                        self.prompt
                        + "\n"
                        + " | ".join(
                            [
                                line["instruction"],
                                line["answer"].replace("<AND>", " <AND> ").replace("<OR>", " <OR> "),
                                model_pred,
                                "",
                            ]
                        )
                    )
                    messages = [
                        {"role": "user", "content": question},
                    ]

                    if id not in grade_results:
                        sample_grade = {"model": [], "content": [], "score": []}
                    else:
                        sample_grade = grade_results[id]

                    grade_sample_run_complete = False
                    temperature = 0.0

                    while not grade_sample_run_complete:
                        try:
                            response = self.client.chat.completions.create(model=self.gpt_model, max_tokens=3, temperature=temperature, messages=messages, timeout=15)
                            content = response["choices"][0]["message"]["content"]
                            flag = True
                            try_time = 1
                            while flag:
                                try:
                                    content = content.split(" ")[0].strip()
                                    score = float(content)
                                    if score > 1.0 or score < 0.0:
                                        assert False
                                    flag = False
                                except:
                                    question = (
                                        self.prompt
                                        + "\n"
                                        + " | ".join(
                                            [
                                                line["instruction"],
                                                line["answer"].replace("<AND>", " <AND> ").replace("<OR>", " <OR> "),
                                                model_pred,
                                                "",
                                            ]
                                        )
                                        + "\nPredict the correctness of the answer (digit): "
                                    )
                                    messages = [
                                        {"role": "user", "content": question},
                                    ]
                                    response = self.client.chat.completions.create(model=self.gpt_model, max_tokens=3, temperature=temperature, messages=messages, timeout=15)
                                    content = response["choices"][0]["message"]["content"]
                                    try_time += 1
                                    temperature += 0.5
                                    print(f"{id} try {try_time} times")
                                    print(content)
                                    if try_time > 5:
                                        score = 0.0
                                        flag = False
                            grade_sample_run_complete = True
                        except Exception as e:
                            # gpt4 may have token rate limit
                            print(e)
                            print("sleep 15s")
                            time.sleep(15)

                    if len(sample_grade["model"]) >= j + 1:
                        sample_grade["model"][j] = response["model"]
                        sample_grade["content"][j] = content
                        sample_grade["score"][j] = score
                    else:
                        sample_grade["model"].append(response["model"])
                        sample_grade["content"].append(content)
                        sample_grade["score"].append(score)
                        sample_grade["query"] = line["instruction"]
                        sample_grade["response"] = model_pred
                        sample_grade["ground_truth"] = line["answer"]
                    grade_results[id] = sample_grade

                    with open(grade_file, "w") as f:
                        json.dump(grade_results, f, indent=4)

        cap_socres = {k: [0.0] * self.num_run for k in self.columns[:-2]}
        self.counter["total"] = self.len_data

        cap_socres2 = {k: [0.0] * self.num_run for k in self.columns2[:-2]}
        counter2 = {self.columns2[i]: self.cap_set_counter[i] for i in range(len(self.cap_set_counter))}
        counter2["total"] = self.len_data

        for k, v in grade_results.items():
            # if sub_set is not None and k not in sub_set:
            #     continue
            for i in range(self.num_run):
                score = v["score"][i]
                caps = set(self.caps[k])
                for c in caps:
                    cap_socres[c][i] += score

                cap_socres["total"][i] += score

                index = self.cap_set_list.index(caps)
                cap_socres2[self.cap_set_names[index]][i] += score
                cap_socres2["total"][i] += score

        for k, v in cap_socres.items():
            cap_socres[k] = np.array(v) / self.counter[k] * 100

        std = round(cap_socres["total"].std(), self.decimal_places)
        total_copy = cap_socres["total"].copy()
        runs = str(list(np.round(total_copy, self.decimal_places)))

        for k, v in cap_socres.items():
            cap_socres[k] = round(v.mean(), self.decimal_places)

        cap_socres["std"] = std
        cap_socres["runs"] = runs
        self.result1.loc[model.name] = cap_socres

        for k, v in cap_socres2.items():
            cap_socres2[k] = round(np.mean(np.array(v) / counter2[k] * 100), self.decimal_places)
        cap_socres2["std"] = std
        cap_socres2["runs"] = runs
        self.result2.loc[model.name] = cap_socres2

        self.result1.to_csv(cap_score_file)
        self.result2.to_csv(cap_int_score_file)

        print(f"cap score saved to {cap_score_file}")
        print(f"cap int score saved to {cap_int_score_file}")
        print("=" * 20)
        print(f"cap score:")
        print(self.result1)
        print("=" * 20)
        print(f"cap int score:")
        print(self.result2)
        print("=" * 20)


if __name__ == "__main__":
    data = MMVetDataset(api_key=None, cache_dir="/data/pufanyi/cache")
