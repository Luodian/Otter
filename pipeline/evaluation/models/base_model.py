from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict

import importlib

AVAILABLE_MODELS: Dict[str, str] = {"video_chat": "VideoChat", "llama_adapter": "LlamaAdapter", "mplug_owl" : "mPlug_owl", "video_chatgpt" : "Video_ChatGPT"}


class BaseModel(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path

    @abstractmethod
    def generate(self, input_data: dict):
        pass


def load_model(model_name: str, model_args: Dict[str, str]) -> BaseModel:
    assert model_name in AVAILABLE_MODELS, f"{model_name} is not an available model."
    module_path = "pipeline.evaluation.models." + model_name
    model_formal_name = AVAILABLE_MODELS[model_name]
    imported_module = importlib.import_module(module_path)
    model_class = getattr(imported_module, model_formal_name)
    print(f"Imported class: {model_class}")
    return model_class(**model_args)
