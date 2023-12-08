import torch
from .video_chatgpt.eval.model_utils import load_video, initialize_model
from .video_chatgpt.inference import video_chatgpt_infer

from .base_model import BaseModel

model_name = "/mnt/lustre/yhzhang/kaichen/video_ChatGPT/LLaVA-Lightening-7B-v1-1."
projection_path = "/mnt/lustre/yhzhang/kaichen/video_ChatGPT/video_chatgpt-7B.bin"


class Video_ChatGPT(BaseModel):
    def __init__(self, model_path: str):
        super().__init__("video_chatgpt", model_path)
        (
            self.model,
            self.vision_tower,
            self.tokenizer,
            self.image_processor,
            self.video_token_len,
        ) = initialize_model(model_name, projection_path)

    def generate(self, input_data: dict):
        video_dir = input_data.get("video_root", "")
        video_frames = load_video(input_data["video_path"])

        object_description = input_data["object_description"]
        if object_description != "None":
            context = f"Given context:{object_description}. "
        else:
            context = ""
        prompts_input = context + input_data["question"]

        output = video_chatgpt_infer(
            video_frames, prompts_input, conv_mode="video-chatgpt_v1", model=self.model, vision_tower=self.vision_tower, tokenizer=self.tokenizer, image_processor=self.image_processor, video_token_len=self.video_token_len
        )
        return output


if __name__ == "__main__":
    model = Video_ChatGPT("")
    device = torch.device("cuda")
    model.model = model.model.to(device)
    model.vision_tower = model.vision_tower.to(device)
    data = {
        "video_idx": "./data_source/multi_hop_reasoning/03f2ed96-1719-427d-acf4-8bf504f1d66d.mp4",
        "question": "What is in this image?",
    }
    print(model.generate(data))
