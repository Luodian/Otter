from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict, List, Any
import base64
import io
import importlib

AVAILABLE_EVAL_DATASETS: Dict[str, str] = {
    "mmbench": "MMBenchDataset",
    "mme": "MMEDataset",
    "mathvista": "MathVistaDataset",
    "mmvet": "MMVetDataset",
    "seedbench": "SEEDBenchDataset",
    "pope": "PopeDataset",
    "scienceqa": "ScienceQADataset",
    "magnifierbench": "MagnifierBenchDataset",
    "mmmu": "MMMUDataset",
}

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


class BaseEvalDataset(ABC):
    def __init__(self, name: str, dataset_path: str, *, max_batch_size: int = 1):
        self.name = name
        self.dataset_path = dataset_path
        self.max_batch_size = max_batch_size

    def evaluate(self, model, **kwargs):
        return self._evaluate(model, **kwargs)
        # batch = min(model.max_batch_size, self.max_batch_size)
        # if batch == 1:
        #     return self._evaluate(model, **kwargs)
        # else:
        #     kwargs["batch"] = batch
        #     return self._evaluate(model, **kwargs)

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
    # get dataset args without "name"
    init_args = dataset_args.copy()
    init_args.pop("name")
    return dataset_class(**init_args)
