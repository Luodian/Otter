from .LLaMA_Adapter.imagebind_LLM.ImageBind import data as data_utils
from .LLaMA_Adapter.imagebind_LLM import llama

from .base_model import BaseModel

import os


llama_dir = "/mnt/petrelfs/share_data/zhangyuanhan/llama_adapter_v2_multimodal"

video_dir = "data_source/multi_hop_reasoning"


class LlamaAdapter(BaseModel):
    # checkpoint will be automatically downloaded
    def __init__(self, model_path: str):
        self.model = llama.load(model_path, llama_dir)
        self.model.eval()

    def generate(self, input_data):
        inputs = {}
        image = data_utils.load_and_transform_video_data([os.path.join(video_dir, input_data["video_idx"] + ".mp4")], device="cuda")
        inputs["Image"] = [image, 1]

        # import pdb;pdb.set_trace()
        results = self.model.generate(inputs, [llama.format_prompt(input_data["question"])], max_gen_len=256)
        result = results[0].strip()
        return result


if __name__ == "__main__":
    model = LlamaAdapter("", "")
    data = {"video_idx": "03f2ed96-1719-427d-acf4-8bf504f1d66d.mp4", "question": "What is in this image?"}
    print(model.generate(data))
