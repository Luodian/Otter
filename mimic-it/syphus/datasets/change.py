"""
This file contains the implementation of the SpotTheDifference and CleverChange datasets.
"""

import importlib
import json

from abstract_dataset import AbstractDataset


class SpotTheDifference(AbstractDataset):
    def __init__(
        self,
        name: str = "SpotTheDifference",
        prompt_path: str = "prompts/spot_the_difference.json",
        query_inputs_path: str = "annotations/spot_the_difference/train.json",
    ):
        super().__init__(name, prompt_path, query_inputs_path)

    def _load_query_inputs(self, path: str) -> list[str]:
        with open(path, "r") as f:
            json_data = json.load(f)
        results = []
        for inner_dict in json_data:
            file_id = inner_dict["img_id"]
            sentences = inner_dict["sentences"]
            results.append(
                {
                    "id": file_id,
                    "sentences": "\n".join(sentences),
                }
            )
        return results


class CocoSpotTheDifference(AbstractDataset):
    def __init__(
        self,
        name: str = "CocoSpotTheDifference",
        prompt_path: str = "prompts.coco_spot_the_difference_prompt",
        query_inputs_path: str = "annotations/coco_spot_the_difference/csd_query.json",
    ):
        super().__init__(name, prompt_path, query_inputs_path)

    def _load_query_inputs(self, path: str) -> list[dict[str, str]]:
        with open(path) as f:
            json_data = json.load(f)
        results = []
        for file_id, inner_dict in json_data.items():
            sentences = inner_dict["sentences"]
            results.append(
                {
                    "id": file_id,
                    "sentences": sentences,
                }
            )
        return results

    def _load_prompt(self, path: str) -> dict[str, str]:
        prompt_file = importlib.import_module(path)
        return {
            "system_message": prompt_file.system_message,
            "in_context": prompt_file.in_context,
        }
