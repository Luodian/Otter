from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict

import importlib

AVAILABLE_EVAL_DATASETS: Dict[str, str] = {
    "mhbench": "MultiHopBenchDataset",
}


class BaseEvalDataset(ABC):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    @abstractmethod
    def evaluate(self, model: str, output_file: str):
        pass


def load_dataset(dataset_name: str, dataset_args: Dict[str, str] = {}) -> BaseEvalDataset:
    assert dataset_name in AVAILABLE_EVAL_DATASETS, f"{dataset_name} is not an available eval dataset."
    module_path = "pipeline.evaluation.eval_datasets." + dataset_name
    dataset_formal_name = AVAILABLE_EVAL_DATASETS[dataset_name]
    imported_module = importlib.import_module(module_path)
    dataset_class = getattr(imported_module, dataset_formal_name)
    print(f"Imported class: {dataset_class}")
    # import pdb;pdb.set_trace()
    return dataset_class(**dataset_args)
