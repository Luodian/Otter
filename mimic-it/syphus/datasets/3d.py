"""
This file contains the implementation of the SceneNavigation and SceneRef and Scene QA datasets.
"""

import json
import random

from abstract_dataset import AbstractDataset


class SceneNavigation(AbstractDataset):
    def __init__(
        self,
        name: str = "SceneNavigation",
        in_context_path: str = "prompts/scene_navigation.json",
        query_inputs_path: str = "annotations/scene_navigation/scan_info.json",
    ):
        super().__init__(name, in_context_path, query_inputs_path)

    def _load_query_inputs(self, path: str) -> list[str]:
        with open(path, "r") as f:
            json_data = json.load(f)
        results = []
        counter = 0
        for scene_id, inner_dict in json_data.items():
            # if counter > 7:
            #     break
            descriptions = inner_dict["description"]
            random.shuffle(descriptions)
            real_descriptions = []
            for cur_description in descriptions[:50]:
                real_descriptions.append(cur_description[1])
            results.append(
                {
                    "id": scene_id,
                    "sentences": "\n".join(real_descriptions),
                }
            )
            counter += 1
        return results
