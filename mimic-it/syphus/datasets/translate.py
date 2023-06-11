import importlib
import json

from abstract_dataset import AbstractDataset


class TranslationDataset(AbstractDataset):
    def __init__(
        self,
        name: str = "Translations",
        prompt_path: str = "prompts.translation_prompt",
        query_inputs_path: str = None,
    ):
        super().__init__(name, prompt_path, query_inputs_path)

    def _load_prompt(self, path: str) -> dict[str, str]:
        prompt_file = importlib.import_module(path)
        return {
            "system_message": prompt_file.system_message,
            "in_context": prompt_file.in_context,
        }

    def _load_query_inputs(self, path: str) -> list[str]:
        with open(path, "r") as f:
            json_data = json.load(f)["data"]
        temp = []
        for file_id, i in json_data.items():
            all_string = "<a>" + i["instruction"] + "</a>" + "<b>" + i["answer"] + "</b>"
            temp.extend([{"id": file_id, "sentences": all_string}])
        return temp
