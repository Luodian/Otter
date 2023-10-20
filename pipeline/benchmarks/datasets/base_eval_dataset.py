from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict

import importlib

AVAILABLE_EVAL_DATASETS: Dict[str, str] = {
    "mhbench": "MultiHopBenchDataset",
    "mmbench": "MMBenchDataset",
    "mme": "MMEDataset",
    "mathvista": "MathVistaDataset",
    "mmvet": "MMVetDataset",
    "seedbench": "SEEDBenchDataset",
    "pope": "PopeDataset",
}


class BaseEvalDataset(ABC):
    def __init__(self, name: str, dataset_path: str, *, can_batch_eval: bool = False):
        self.name = name
        self.dataset_path = dataset_path
        self.can_batch_eval = can_batch_eval

    def check_batch(self, batch, model):
        if batch == 1:
            return batch
        if not self.can_batch_eval:
            batch = 1
            print(f"Dataset {self.name} does not support batch evaluation. Batch size is set to 1.")
        elif not model.can_batch_generate:
            batch = 1
            print(f"Model {model.name} does not support batch evaluation. Batch size is set to 1.")
        return batch

    def evaluate(self, model, **kwargs):
        batch = kwargs.get("batch", 1)
        batch = self.check_batch(batch, model)
        kwargs["batch"] = batch
        return self._evaluate(model, **kwargs)

    @abstractmethod
    def _evaluate(self, model: str):
        pass


def load_dataset(dataset_name: str, dataset_args: Dict[str, str] = {}) -> BaseEvalDataset:
    assert dataset_name in AVAILABLE_EVAL_DATASETS, f"{dataset_name} is not an available eval dataset."
    module_path = "pipeline.benchmarks.datasets." + dataset_name
    dataset_formal_name = AVAILABLE_EVAL_DATASETS[dataset_name]
    imported_module = importlib.import_module(module_path)
    dataset_class = getattr(imported_module, dataset_formal_name)
    print(f"Imported class: {dataset_class}")
    # import pdb;pdb.set_trace()
    return dataset_class(**dataset_args)
