import base64
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from .base_eval_dataset import BaseEvalDataset


class MMBenchDataset(BaseEvalDataset):
    def __init__(self, data_path="Otter-AI/MMBench", *, sys_prompt="There are several options:", version="20230712", split="validation", cache_dir=None, default_output_path="."):
        super().__init__("MMBenchDataset", data_path)
        version = str(version)
        self.df = load_dataset("Otter-AI/MMBench", version, split=split, cache_dir=cache_dir).to_pandas()
        self.default_output_path = default_output_path
        self.sys_prompt = sys_prompt

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None

    def get_data(self, idx):
        index = self.df.iloc[idx]["index"]
        image = self.df.iloc[idx]["image"]
        question = self.df.iloc[idx]["question"]
        answer = self.df.iloc[idx]["answer"] if "answer" in self.df.iloc[0].keys() else None
        catetory = self.df.iloc[idx]["category"]
        l2_catetory = self.df.iloc[idx]["l2-category"]

        option_candidate = ["A", "B", "C", "D", "E"]
        options = {cand: self.load_from_df(idx, cand) for cand in option_candidate if self.load_from_df(idx, cand) is not None}
        options_prompt = f"{self.sys_prompt}\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"

        hint = self.load_from_df(idx, "hint")
        data = {
            "img": image,
            "question": question,
            "answer": answer,
            "options": options_prompt,
            "category": catetory,
            "l2-category": l2_catetory,
            "options_dict": options,
            "index": index,
            "context": hint,
        }
        return data

    def evaluate(self, model):
        output_file = os.path.join(self.default_output_path, f"{model.name}_mmbench_eval_result.xlsx")

        results = []

        for idx in tqdm(range(len(self.df))):
            cur_data = self.get_data(idx)
            image = cur_data["img"]
            question = cur_data["question"]
            answer = cur_data["answer"]
            options = cur_data["options"]
            contexts = cur_data["context"]
            index = cur_data["index"]
            options_dict = cur_data["options_dict"]
            category = cur_data["category"]
            l2_category = cur_data["l2-category"]
            cur_prompt = contexts + " " + question + " " + options if contexts is not None else question + " " + options
            pred_answer = model.generate(cur_prompt, image)
            # print(f"model response: {pred_answer}")
            result = dict()
            result["question"] = question
            result["answer"] = answer
            result.update(options_dict)
            result["prediction"] = pred_answer
            if category is not None:
                result["category"] = category
            if l2_category is not None:
                result["l2-category"] = l2_category
            result["index"] = index
            results.append(result)

        df = pd.DataFrame(results)
        with pd.ExcelWriter(
            output_file,
            engine="xlsxwriter",
        ) as writer:
            df.to_excel(writer, index=False)

        print(f"MMBench Evaluator: Result saved to {output_file}.")
