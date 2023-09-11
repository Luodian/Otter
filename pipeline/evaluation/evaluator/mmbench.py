import base64
import io

import pandas as pd
from mmengine.dataset import Compose
from PIL import Image
from tqdm import tqdm


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


class MMBenchEvaluator(object):
    def __init__(self, data_file, sys_prompt="There are several options:"):
        df = pd.read_csv(data_file, sep="\t")
        self.default_output_file = data_file.replace(".tsv", "_eval_result.xlsx")
        self.sys_prompt = sys_prompt
        self.data = []
        for idx, item in df.iterrows():
            index = item["index"]
            image = item["image"]
            image = decode_base64_to_image(image)
            question = item["question"]
            answer = item["answer"] if "answer" in df.iloc[0].keys() else None
            catetory = item["category"]
            l2_catetory = item["l2-category"]

            option_candidate = ["A", "B", "C", "D", "E"]
            options = {}
            for cand in option_candidate:
                if cand in item and not pd.isna(item[cand]):
                    options[cand] = item[cand]
            options_prompt = f"{self.sys_prompt}\n"
            for key, item in options.items():
                options_prompt += f"{key}. {item}\n"

            if "hint" in item and not pd.isna(item["hint"]):
                hint = item["hint"]
            else:
                hint = None

            self.data.append(
                {
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
            )

    def evaluate(self, model, output_file=None):
        if output_file is None:
            output_file = self.default_output_file

        results = []

        for cur_data in tqdm(self.data):
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
            print(f"model response: {pred_answer}")
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
