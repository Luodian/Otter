from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict

import importlib

AVAILABLE_MODELS: Dict[str, str] = {
    "video_chat": "VideoChat",
    "otter_video": "OtterVideo",
    "llama_adapter": "LlamaAdapter",
    "mplug_owl": "mPlug_owl",
    "video_chatgpt": "Video_ChatGPT",
    "otter_image": "OtterImage",
    "frozen_bilm": "FrozenBilm",
    "idefics": "Idefics",
    "fuyu": "Fuyu",
    "otterhd": "OtterHD",
    "instructblip": "InstructBLIP",
    "qwen_vl": "QwenVL",
    "llava_model": "LLaVA_Model",
    "instructblip": "InstructBLIP",
    "gpt4v": "OpenAIGPT4Vision",
}


class BaseModel(ABC):
    def __init__(self, model_name: str, model_path: str, *, max_batch_size: int = 1):
        self.name = model_name
        self.model_path = model_path
        self.max_batch_size = max_batch_size

    @abstractmethod
    def generate(self, **kwargs):
        pass

    @abstractmethod
    def eval_forward(self, **kwargs):
        pass


def load_model(model_name: str, model_args: Dict[str, str]) -> BaseModel:
    assert model_name in AVAILABLE_MODELS, f"{model_name} is not an available model."
    module_path = "pipeline.benchmarks.models." + model_name
    model_formal_name = AVAILABLE_MODELS[model_name]
    imported_module = importlib.import_module(module_path)
    model_class = getattr(imported_module, model_formal_name)
    print(f"Imported class: {model_class}")
    model_args.pop("name")
    return model_class(**model_args)
