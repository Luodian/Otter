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

import time
import requests

utc_plus_8 = pytz.timezone("Asia/Singapore")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)


def get_chat_response(promot, api_key, model="gpt-4-0613", temperature=0, max_tokens=256, n=1, patience=5, sleep_time=5):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Your task is to judge whether the model response is correct to answer the given question or not."},
        {"role": "user", "content": promot},
    ]

    payload = {"model": model, "messages": messages}

    while patience > 0:
        patience -= 1
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30,
            )
            response.raise_for_status()
            response_data = response.json()

            prediction = response_data["choices"][0]["message"]["content"].strip()
            if prediction != "" and prediction is not None:
                return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(e)
            time.sleep(sleep_time)

    return ""


def prepare_query(model_answer_item, api_key):
    freeform_question = model_answer_item["freeform_question"]
    freeform_response = model_answer_item["freeform_response"]
    correct_answer = model_answer_item["freeform_answer"]

    # Formulating the prompt for ChatGPT
    prompt = f"Question: {freeform_question}\nModel Response: {freeform_response}\nGround Truth: {correct_answer}\nWill the model response be considered correct? You should only answer yes or no."

    # Querying ChatGPT
    chat_response = get_chat_response(prompt, api_key)

    return chat_response


class MagnifierBenchDataset(BaseEvalDataset):
    def __init__(
        self,
        data_path: str = "Otter-AI/MagnifierBench",
        *,
        cache_dir: Union[str, None] = None,
        default_output_path: str = "./logs/MagBench",
        split: str = "test",
        debug: bool = False,
        prompt="",
        api_key=None,
    ):
        super().__init__("MagnifierBench", data_path)

        self.default_output_path = default_output_path
        if not os.path.exists(self.default_output_path):
            os.makedirs(self.default_output_path)

        self.cur_datetime = utc_plus_8_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.data = load_dataset(data_path, split=split, cache_dir=cache_dir, revision="main")
        self.debug = debug
        self.prompt = prompt
        self.api_key = api_key

    def parse_pred_ans(self, pred_ans, question):
        match = re.search(r"The answer is ([A-D])", pred_ans)
        if match:
            return match.group(1)
        choices = ["A", "B", "C", "D"]
        for selection in choices:
            if selection in pred_ans:
                return selection
        pattern = "A\\. (.+?), B\\. (.+?), C\\. (.+?), D\\. (.+)"
        matches = re.search(pattern, question)
        if matches:
            options = {"A": matches.group(1), "B": matches.group(2), "C": matches.group(3), "D": matches.group(4)}
            for c, option in options.items():
                option = option.strip()
                if option.endswith(".") or option.endswith(",") or option.endswith("?"):
                    option = option[:-1]
                if option.upper() in pred_ans.upper():
                    return c
        for selection in choices:
            if selection in pred_ans.upper():
                return selection
        return "other"

    def _evaluate(self, model):
        model_score_dict = {}

        # output_path = os.path.join(self.default_output_path, f"{model.name}_{self.cur_datetime}")
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        # model_path: str = "Salesforce/instructblip-vicuna-7b"
        model_version = model.name.split("/")[-1]
        model_answer_path = os.path.join(self.default_output_path, f"{model_version}_{self.cur_datetime}_answer.json")
        result_path = os.path.join(self.default_output_path, f"{model_version}_{self.cur_datetime}_score.json")
        model_answer = {}

        score = 0
        num_data = 0

        ff_score = 0

        for data in tqdm(self.data, desc="Evaluating", total=len(self.data)):
            question = f"{self.prompt} {data['instruction']}" if self.prompt else data["instruction"]
            if len(data["images"]) != 1:
                print(f"Warning: {data['id']} has {len(data['images'])} images.")
                print(f"Skipping {data['id']}")
                continue

            model_response = model.generate(question, data["images"][0])

            pred_ans = self.parse_pred_ans(model_response, question)

            freeform_question = (question.split("?")[0] + "?").replace(self.prompt, "").strip()
            options = question.split("?")[1]
            answer_option = data["answer"]
            for single_opt in options.split(","):
                single_opt = single_opt.strip()
                if single_opt.startswith(answer_option.upper()):
                    freeform_answer = single_opt.split(".")[1].strip()
                    break

            ff_response = model.generate(freeform_question, data["images"][0])
            if self.debug:
                print(f"Question: {question}")
                print(f"Answer: {data['answer']}")
                print(f"Raw prediction: {model_response}")
                print(f"Parsed prediction: {pred_ans}\n")
                print(f"Freeform question: {freeform_question}")
                print(f"Freeform answer: {freeform_answer}")
                print(f"Freeform response: {ff_response}\n")

            num_data += 1
            if pred_ans == data["answer"]:
                score += 1
            model_answer[data["id"]] = {
                "question": question,
                "options": options,
                "model_response": model_response,
                "parsed_output": pred_ans,
                "answer": data["answer"],
                "freeform_question": freeform_question,
                "freeform_response": ff_response,
                "freeform_answer": freeform_answer,
            }
            with open(model_answer_path, "w") as f:
                json.dump(model_answer, f, indent=2)

        model_score_dict["score"] = score
        model_score_dict["total"] = len(self.data)
        model_score_dict["accuracy"] = score / len(self.data)

        print(f"Start query GPT-4 for free-form evaluation...")
        for data_id in tqdm(model_answer.keys(), desc="Querying GPT-4"):
            model_answer_item = model_answer[data_id]
            gpt_response = prepare_query(model_answer_item, self.api_key)
            if gpt_response.lower() == "yes":
                ff_score += 1
            elif gpt_response.lower() == "no":
                ff_score += 0
            else:
                print(f"Warning: {data_id} has invalid GPT-4 response: {gpt_response}")
                print(f"Skipping {data_id}")
                continue

        model_score_dict["freeform_score"] = ff_score
        model_score_dict["freeform_accuracy"] = ff_score / len(model_answer)

        with open(result_path, "w") as f:
            json.dump(model_score_dict, f, indent=2)

        print(f"Model answer saved to {model_answer_path}")
        print(f"Model score saved to {result_path}")
        print(json.dumps(model_score_dict, indent=2))

        return model_score_dict
