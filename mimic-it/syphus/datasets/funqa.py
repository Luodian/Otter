"""
This file contains the implementation of the FunQA datasets.
"""

import json
from abstract_dataset import AbstractDataset


class FunQA_translation(AbstractDataset):
    def __init__(
        self,
        name: str = "FunQA_translation",
        prompt_path: str = "prompts/funqa_translation.json",
        query_inputs_path: str = "annotations/funqa_annotation/annotations_tr.json",
    ):
        super().__init__(name, prompt_path, query_inputs_path)

    def _load_query_inputs(self, path: str) -> list[dict[str, str]]:
        with open(path, "r") as f:
            json_data = json.load(f)
        q_dic = {
            "H1": "Find the video" "s humorous moment.",
            "H2": "Description of the video" "s humorous moment.",
            "H3": "Why is the whole video humorous.",
            "H4": "Please provide a caption for the video.",
            "C1": "Find the video" "s creative moment.",
            "C2": "Description of the video" "s creative moment.",
            "C3": "Why is the whole video creative.",
            "C4": "Please provide a caption for the video.",
            "C5": "Please score the video" "s creativity in [0-20].",
            "M1": "Find the video" "s magic moment.",
            "M2": "Description of the video" "s magic moment.",
            "M3": "Why is the whole video magic.",
        }
        query_inputs = []
        for item in json_data:
            task = list(q_dic)[list(q_dic.values()).index(item["instruction"])]
            if task[-1] == "1" or task[-1] == "5":
                continue
            query_inputs.append(
                {
                    "id": item["visual_input"].split("/")[-1] + "_" + task,
                    "sentences": item["output"],
                }
            )
        return query_inputs


class FunQA_mcqa(AbstractDataset):
    def __init__(
        self,
        name: str = "DenseCaptions",
        prompt_path: str = "prompts/funqa_mcqa.json",
        query_inputs_path: str = "annotations/funqa_annotation/annotations_mcqa.json",
    ):
        super().__init__(name, prompt_path, query_inputs_path)

    def _load_query_inputs(self, path: str) -> list[dict[str, str]]:
        with open(path, "r") as f:
            json_data = json.load(f)
        q_dic = {
            "H1": "Find the video" "s humorous moment.",
            "H2": "Description of the video" "s humorous moment.",
            "H3": "Why is the whole video humorous.",
            "H4": "Please provide a caption for the video.",
            "C1": "Find the video" "s creative moment.",
            "C2": "Description of the video" "s creative moment.",
            "C3": "Why is the whole video creative.",
            "C4": "Please provide a caption for the video.",
            "C5": "Please score the video" "s creativity in [0-20].",
            "M1": "Find the video" "s magic moment.",
            "M2": "Description of the video" "s magic moment.",
            "M3": "Why is the whole video magic.",
        }
        query_inputs = []

        for item in json_data:
            task = list(q_dic)[list(q_dic.values()).index(item["instruction"])]
            if task[-1] == "2":
                now_query_input = "description: " + item["output"] + "\n"
                continue
            elif task[-1] == "3":
                now_query_input += "counter-intuitive reason: " + item["output"]
                query_inputs.append(
                    {
                        "id": item["visual_input"].split("/")[-1],
                        "sentences": now_query_input,
                    }
                )
        return query_inputs


class FunQA_dia(AbstractDataset):
    def __init__(
        self,
        name: str = "DenseCaptions",
        prompt_path: str = "prompts/funqa_dia.json",
        query_inputs_path: str = "annotations/funqa_annotation/annotations_dia.json",
    ):
        super().__init__(name, prompt_path, query_inputs_path)

    def _load_query_inputs(self, path: str) -> list[dict[str, str]]:
        with open(path, "r") as f:
            json_data = json.load(f)
        q_dic = {
            "H1": "Find the video" "s humorous moment.",
            "H2": "Description of the video" "s humorous moment.",
            "H3": "Why is the whole video humorous.",
            "H4": "Please provide a caption for the video.",
            "C1": "Find the video" "s creative moment.",
            "C2": "Description of the video" "s creative moment.",
            "C3": "Why is the whole video creative.",
            "C4": "Please provide a caption for the video.",
            "C5": "Please score the video" "s creativity in [0-20].",
            "M1": "Find the video" "s magic moment.",
            "M2": "Description of the video" "s magic moment.",
            "M3": "Why is the whole video magic.",
        }
        query_inputs = []

        for item in json_data:
            task = list(q_dic)[list(q_dic.values()).index(item["instruction"])]
            if task[-1] == "2":
                now_query_input = "description: " + item["output"] + "\n"
                continue
            elif task[-1] == "3":
                now_query_input += "counter-intuitive reason: " + item["output"]
                query_inputs.append(
                    {
                        "id": item["visual_input"].split("/")[-1],
                        "sentences": now_query_input,
                    }
                )
        return query_inputs
