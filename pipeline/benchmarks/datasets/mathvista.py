import base64
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from .base_eval_dataset import BaseEvalDataset
import json
from io import BytesIO
import pytz
import datetime
import openai
import time
import re
import io
from Levenshtein import distance

utc_plus_8 = pytz.timezone("Asia/Singapore")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)

demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""


import time
import requests
import json
import ast


def get_chat_response(promot, api_key, model="gpt-3.5-turbo", temperature=0, max_tokens=256, n=1, patience=5, sleep_time=5):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
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


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(response, problem, quick_extract=False, api_key=None, pid=None, gpt_model="gpt-4-0613"):
    question_type = problem["question_type"]
    answer_type = problem["answer_type"]
    choices = problem["choices"]
    query = problem["query"]

    if response == "":
        return ""

    if question_type == "multi_choice" and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass

    # quick extraction
    if quick_extract:
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except:
            pass

    else:
        # general extraction
        try:
            full_prompt = create_test_prompt(demo_prompt, query, response)
            extraction = get_chat_response(full_prompt, api_key=api_key, model=gpt_model, n=1, patience=5, sleep_time=5)
            return extraction
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {pid}")

    return ""


def get_acc_with_contion(res_pd, key, value):
    if key == "skills":
        # if value in res_pd[key]:
        total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
    else:
        total_pd = res_pd[res_pd[key] == value]

    correct_pd = total_pd[total_pd["true_false"] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return len(correct_pd), len(total_pd), acc


def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]
    # return min(choices, key=lambda choice: distance(prediction, choice))


def normalize_extracted_answer(extraction, choices, question_type, answer_type, precision):
    """
    Normalize the extracted answer to match the answer type
    """
    if question_type == "multi_choice":
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except:
                extraction = ""

        # extract "A" from "(A) text"
        letter = re.findall(r"\(([a-zA-Z])\)", extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        options = [chr(ord("A") + i) for i in range(len(choices))]

        if extraction in options:
            # convert option letter to text, e.g. "A" -> "text"
            ind = options.index(extraction)
            extraction = choices[ind]
        else:
            # select the most similar option
            extraction = get_most_similar(extraction, choices)
        assert extraction in choices

    elif answer_type == "integer":
        try:
            extraction = str(int(float(extraction)))
        except:
            extraction = None

    elif answer_type == "float":
        try:
            extraction = str(round(float(extraction), precision))
        except:
            extraction = None

    elif answer_type == "list":
        try:
            extraction = str(extraction)
        except:
            extraction = None

    return extraction


def get_pil_image(raw_image_data) -> Image.Image:
    if isinstance(raw_image_data, Image.Image):
        return raw_image_data

    elif isinstance(raw_image_data, dict) and "bytes" in raw_image_data:
        return Image.open(io.BytesIO(raw_image_data["bytes"]))

    elif isinstance(raw_image_data, str):  # Assuming this is a base64 encoded string
        image_bytes = base64.b64decode(raw_image_data)
        return Image.open(io.BytesIO(image_bytes))

    else:
        raise ValueError("Unsupported image data format")


def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False


class MathVistaDataset(BaseEvalDataset):
    def __init__(
        self,
        data_path="Otter-AI/MathVista",
        split="test",
        default_output_path="./logs/MathVista",
        cache_dir=None,
        api_key=None,
        gpt_model="gpt-4-0613",
        debug=False,
        quick_extract=False,
    ):
        super().__init__("MathVistaDataset", data_path)
        name_converter = {"dev": "validation", "test": "test"}
        self.data = load_dataset("Otter-AI/MathVista", split=name_converter[split], cache_dir=cache_dir).to_pandas()
        if debug:
            self.data = self.data.sample(5)
        # data_path = "/home/luodian/projects/Otter/archived/testmini_image_inside.json"
        # with open(data_path, "r", encoding="utf-8") as f:
        #     self.data = json.load(f)

        self.debug = debug
        self.quick_extract = quick_extract

        self.default_output_path = default_output_path
        if os.path.exists(self.default_output_path) is False:
            os.makedirs(self.default_output_path)
        self.cur_datetime = utc_plus_8_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.api_key = api_key
        self.gpt_model = gpt_model

    def create_query(self, problem, shot_type):
        ### [2] Test query
        # problem info
        question = problem["question"]
        unit = problem["unit"]
        choices = problem["choices"]
        precision = problem["precision"]
        question_type = problem["question_type"]
        answer_type = problem["answer_type"]

        # hint
        if shot_type == "solution":
            if question_type == "multi_choice":
                assert answer_type == "text"
                hint_text = f"Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
            else:
                assert answer_type in ["integer", "float", "list"]
                if answer_type == "integer":
                    hint_text = f"Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."

                elif answer_type == "float" and str(precision) == "1":
                    hint_text = f"Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."

                elif answer_type == "float" and str(precision) == "2":
                    hint_text = f"Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."

                elif answer_type == "list":
                    hint_text = f"Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."
        else:
            assert shot_type == "code"
            hint_text = "Please generate a python code to solve the problem"

        # question
        question_text = f"Question: {question}"
        if unit:
            question_text += f" (Unit: {unit})"

        # choices
        if choices is not None and len(choices) != 0:
            # choices: (A) 1.2 (B) 1.3 (C) 1.4 (D) 1.5
            texts = ["Choices:"]
            for i, choice in enumerate(choices):
                texts.append(f"({chr(ord('A')+i)}) {choice}")
            choices_text = "\n".join(texts)
        else:
            choices_text = ""

        # prompt
        if shot_type == "solution":
            prompt = "Solution: "
        else:
            assert shot_type == "code"
            prompt = "Python code: "

        elements = [hint_text, question_text, choices_text, prompt]
        query = "\n".join([e for e in elements if e != ""])

        query = query.strip()
        return query

    def _evaluate(self, model):
        output_file = os.path.join(self.default_output_path, f"{model.name}_mathvista_eval_submit_{self.cur_datetime}.json")  # directly match Lu Pan's repo format e.g. output_bard.json

        results = {}

        print(f"Number of test problems in total: {len(self.data)}")
        for idx_key, query_data in tqdm(self.data.iterrows(), desc=f"Evaluating {model.name}", total=len(self.data)):
            # query_data = self.data[idx_key]
            results[idx_key] = {}
            results[idx_key].update(query_data)
            if results[idx_key]["choices"] is not None:
                results[idx_key]["choices"] = list(results[idx_key]["choices"])
            results[idx_key].pop("image")
            # problem = query_data["problem"]
            query = self.create_query(problem=query_data, shot_type="solution")
            base64_image = query_data["image"]
            # image = Image.open(BytesIO(base64.b64decode(base64_image)))
            image = get_pil_image(base64_image)
            response = model.generate(query, image)
            if self.debug:
                print(f"\n# Query: {query}")
                print(f"\n# Response: {response}")
            results[idx_key].update({"query": query})
            results[idx_key].update({"response": response})

        with open(output_file, "w") as outfile:
            json.dump(results, outfile)

        results = json.load(open(output_file, "r"))

        print(f"MathVista Evaluator: Results saved to {output_file}")

        for idx_key, row in tqdm(self.data.iterrows(), desc=f"Extracting answers from {model.name}", total=len(self.data)):
            idx_key = str(idx_key)
            response = results[idx_key]["response"]
            extraction = extract_answer(
                response,
                results[idx_key],
                quick_extract=self.quick_extract,
                api_key=self.api_key,
                pid=idx_key,
                gpt_model=self.gpt_model,
            )
            results[idx_key].update({"extraction": extraction})
            answer = results[idx_key]["answer"]
            choices = results[idx_key]["choices"]
            question_type = results[idx_key]["question_type"]
            answer_type = results[idx_key]["answer_type"]
            precision = results[idx_key]["precision"]
            extraction = results[idx_key]["extraction"]

            prediction = normalize_extracted_answer(extraction, choices, question_type, answer_type, precision)
            true_false = safe_equal(prediction, answer)

            results[idx_key]["prediction"] = prediction
            results[idx_key]["true_false"] = true_false

        full_pids = list(results.keys())
        ## [2] Calculate the average accuracy
        total = len(full_pids)
        correct = 0
        for pid in full_pids:
            if results[pid]["true_false"]:
                correct += 1
        accuracy = str(round(correct / total * 100, 2))
        print(f"\nCorrect: {correct}, Total: {total}, Accuracy: {accuracy}%")

        scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}
        ## [3] Calculate the fine-grained accuracy scores
        # merge the 'metadata' attribute into the data
        success_parse = True
        try:
            for pid in results:
                cur_meta = results[pid]["metadata"]
                cur_meta_dict = ast.literal_eval(cur_meta)
                results[pid].update(cur_meta_dict)
        except:
            success_parse = False
            # results[pid].update(results[pid].pop("metadata"))

        # convert the data to a pandas DataFrame
        df = pd.DataFrame(results).T

        print("Number of test problems:", len(df))
        # assert len(df) == 1000 # Important!!!

        if success_parse:
            # asign the target keys for evaluation
            target_keys = [
                "question_type",
                "answer_type",
                "language",
                "source",
                "category",
                "task",
                "context",
                "grade",
                "skills",
            ]

            for key in target_keys:
                print(f"\nType: [{key}]")
                # get the unique values of the key
                if key == "skills":
                    # the value is a list
                    values = []
                    for i in range(len(df)):
                        values += df[key][i]
                    values = list(set(values))
                else:
                    values = df[key].unique()
                # calculate the accuracy for each value
                scores[key] = {}
                for value in values:
                    correct, total, acc = get_acc_with_contion(df, key, value)
                    if total > 0:
                        print(f"[{value}]: {acc}% ({correct}/{total})")
                        scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}

                # sort the scores by accuracy
                scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]["accuracy"]), reverse=True))

        # save the scores
        scores_file = os.path.join(self.default_output_path, f"{model.name}_mathvista_eval_score_{self.cur_datetime}.json")
        print(f"MathVista Evaluator: Score results saved to {scores_file}...")
        with open(scores_file, "w") as outfile:
            json.dump(scores, outfile)
