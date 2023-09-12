from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict

import importlib

AVAILABLE_MODELS: Dict[str, str] = {
    "otter": "Otter",
    "idefics": "Idefics",
    "idefics_otter": "IdeficsOtter",
    "llama_adapter":"LlamaAdapter"
}


class Model(ABC):
    def __init__(self, name: str, model_path: str):
        self.name = name
        self.model_path = model_path

    @abstractmethod
    def generate(self, question: str, raw_image_data: Image.Image):
        pass


def load_model(model_name: str, dataset_args: Dict[str, str]) -> Model:
    assert model_name in AVAILABLE_MODELS, f"{model_name} is not an available model."
    module_path = "pipeline.evaluation.models." + model_name
    model_formal_name = AVAILABLE_MODELS[model_name]
    imported_module = importlib.import_module(module_path)
    dataset_class = getattr(imported_module, model_formal_name)
    print(f"Imported class: {dataset_class}")
    return dataset_class(**dataset_args)
