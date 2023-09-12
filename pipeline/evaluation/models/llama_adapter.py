from LLaMA-Adapter.imagebind_LLM.ImageBind import data as data_utils
from LLaMA-Adapter.imagebind_LLM import llama

from .model import Model


llama_dir = "/mnt/petrelfs/share_data/zhangyuanhan/llama_adapter_v2_multimodal"

video_dir = "pipeline/evaluation/data_source/multi_hop_reasoning"


class LlamaAdapter(Model):
    # checkpoint will be automatically downloaded
    def __init__(self, name: str, model_path: str):
        self.model = llama.load("7B", llama_dir, knn=True)
        self.model.eval()

    def generate(self, data):
        inputs = {}
        image = data_utils.load_and_transform_video_data([os.path.join(video_dir,cur_data["video_idx"])], device='cuda')
        inputs['Image'] = [image, 1]

        results = model.generate(
            inputs,
            [llama.format_prompt(cur_data["question"])],
            max_gen_len=256
        )
        result = results[0].strip()
        return result

if __name__ == "__main__":
    model = LlamaAdapter()
    data = {
        "video_idx": "03f2ed96-1719-427d-acf4-8bf504f1d66d.mp4"
        "question": "What is in this image?"
    }
    print(model.generate("What is in this image?", Image.open("/data/pufanyi/project/Otter-2/pipeline/evaluation/test_data/test.jpg")))