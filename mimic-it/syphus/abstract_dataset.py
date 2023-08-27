from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import importlib
import json

AVAILABLE_DATASETS: List[str] = [
    "change.SpotTheDifference",
    "change.CocoSpotTheDifference",
    "video.DenseCaptions",
    "video.TVCaptions",
    "video.VisualStoryTelling",
    "3d.SceneNavigation",
    "funqa.FunQA_translation",
    "funqa.FunQA_mcqa",
    "funqa.FunQA_dia",
    "fpv.EGO4D",
    "translate.Translation",
]


class AbstractDataset(ABC):
    def __init__(self, name: str, prompt_path: str, query_inputs_path: str):
        """Constructor."""
        self.name: str = name
        self.prompt: Dict[str, Union[str, List[Dict[str, Union[str, List[Dict[str, str]]]]]]] = self._load_prompt(prompt_path)
        self.query_inputs: List[str] = self._load_query_inputs(query_inputs_path)

    def _load_prompt(self, path: str) -> Dict[str, Union[str, List[Dict[str, Union[str, List[Dict[str, str]]]]]]]:
        with open(path, "r") as f:
            json_data: Dict[str, Any] = json.load(f)
        in_context: List[Dict[str, Union[str, List[Dict[str, str]]]]] = json_data["in_context"].copy()
        for n, conv in enumerate(json_data["in_context"]):
            role, content = conv["role"], conv["content"]
            # we need to convert the QA pair into a string
            if role == "assistant":
                content_string = ""
                if isinstance(content, str):
                    content_string = content
                else:
                    for qa_pair in content:
                        for prefix, text in qa_pair.items():
                            content_string += prefix + ": " + text + "\n"
            elif role == "user":
                content_string = content
            else:
                raise ValueError("wrong role. Only user and assistant are allowed.")
            in_context[n] = {"role": role, "content": content_string}
        results: Dict[str, Union[str, List[Dict[str, Union[str, List[Dict[str, str]]]]]]] = {
            "system_message": json_data["system_message"],
            "in_context": in_context,
        }
        return results

    @abstractmethod
    def _load_query_inputs(self, path: str) -> List[str]:
        """
        Load the query_inputs from the given path.
        """
        pass

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Return the item at the given index as a dictionary.
        """
        return self.data[index]

    def __iter__(self) -> "AbstractDataset":
        self.index = 0
        return self

    def __next__(
        self,
    ) -> Dict[str, Union[str, List[Dict[str, Union[str, List[Dict[str, str]]]]]]]:
        if self.index < len(self.query_inputs):
            outputs = {
                "system_messages": self.prompt["system_message"],
                "in_context": self.prompt["in_context"],
                "query_input": self.query_inputs[self.index],
            }
            self.index += 1
            return outputs
        raise StopIteration

    def __len__(self) -> int:
        return len(self.query_inputs)

    def __str__(self) -> str:
        return f"{self.name} dataset"


def get_dataset_by_path(path: str, dataset_args: dict[str, str]) -> AbstractDataset:
    assert path in AVAILABLE_DATASETS, f"{path} is not an available dataset."
    module_path, dataset_name = path.split(".")
    module_path = "datasets." + module_path

    # Import the module and load the class
    imported_module = importlib.import_module(module_path)
    dataset_class = getattr(imported_module, dataset_name)

    # TODO:remove later, Print the imported class for debugging
    print(f"Imported class: {dataset_class}")

    # Instantiate the class and return the instance
    return dataset_class(**dataset_args)


def get_available_datasets() -> List[str]:
    return AVAILABLE_DATASETS
