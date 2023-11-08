import os

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from .base_model import BaseModel

default_path = "Qwen/Qwen-VL-Chat"


class QwenVL(BaseModel):
    def __init__(self, model_name: str = "qwen_vl", model_path: str = default_path):
        super().__init__(model_name, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        self.temp_dir = ".log/temp"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def generate(self, text_prompt: str, raw_image_data: str):
        image_path = os.path.join(self.temp_dir, "temp.jpg")
        raw_image_data.save(image_path)
        query = []
        query.append({"image": image_path})
        query.append({"text": text_prompt})
        query = self.tokenizer.from_list_format(query)
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response

    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for QwenVL
        pass
