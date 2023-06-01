import mimetypes
import os
from io import BytesIO
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
import sys
sys.path.append("/mnt/petrelfs/zhangyuanhan/Otter")
from otter.modeling_otter import OtterForConditionalGeneration


# Disable warnings
requests.packages.urllib3.disable_warnings()

# ------------------- Utility Functions -------------------


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image and Video Handling Functions -------------------


def extract_frames(video_path, num_frames=16):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert("RGB")
            frames.append(frame)

    video.release()
    return frames


def get_image(url: str) -> Union[Image.Image, list]:
    if "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    elif "video" in content_type:
        video_path = "temp_video.mp4"
        if "://" not in url:  # Local file
            video_path = url
        else:  # Remote URL
            with open(video_path, "wb") as f:
                f.write(requests.get(url, stream=True, verify=False).content)
        frames = extract_frames(video_path)
        if "://" in url:  # Only remove the temporary video file if it was downloaded
            os.remove(video_path)
        return frames
    else:
        raise ValueError("Invalid content type. Expected image or video.")


# ------------------- OTTER Prompt and Response Functions -------------------


def get_formatted_prompt(prompt: str, in_context_prompts: list = []) -> str:
    in_context_string = ""
    for in_context_prompt, in_context_answer in in_context_prompts:
        in_context_string += f"<image>User: {in_context_prompt} GPT:<answer> {in_context_answer}<|endofchunk|>"
    return f"{in_context_string}<image>User: {prompt} GPT:<answer>"

# def get_response(
#     encoded_frames: torch.Tensor, prompt: str, model=None, image_processor=None, in_context_prompts: list = []
# ) -> str:
def get_response(image_list, prompt: str, model=None, image_processor=None, in_context_prompts: list = []) -> str:
    input_data = image_list

    if isinstance(input_data, Image.Image):
        vision_x = (
            image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        )
    elif isinstance(input_data, list):  # list of video frames
        patch_images = torch.tensor([])
        for cur_image in input_data:
            cur_vision_x = image_processor.preprocess([cur_image], return_tensors="pt")["pixel_values"]
            if len(patch_images) == 0:
                patch_images = cur_vision_x
            else:
                patch_images = torch.cat((patch_images,cur_vision_x))
        vision_x = patch_images.unsqueeze(1).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt, in_context_prompts),
        ],
        return_tensors="pt",
    )
    
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x["input_ids"].to(model.device),
        attention_mask=lang_x["attention_mask"].to(model.device),
        max_new_tokens=2048,
        # num_beams=3,
        # no_repeat_ngram_size=3,
    )
    # import pdb;pdb.set_trace()
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[-1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output


# ------------------- Main Function -------------------

if __name__ == "__main__":
    model = OtterForConditionalGeneration.from_pretrained(
        "/mnt/petrelfs/share_data/libo/otter9B_LA_incontext2", device_map="auto"
    )
    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    model.eval()

    while True:
        # input_urls = input("Enter the URLs of the images/videos separated by commas (or type 'qt' to exit): ")
        # if input_urls.lower() == "qt":
        #     break

        # urls = [url.strip() for url in input_urls.split(",")]
        urls = [
            "/mnt/petrelfs/share_data/basemodel/dataset/multimodality/coco/train2017/000000339543.jpg",
            "/mnt/petrelfs/share_data/basemodel/dataset/multimodality/coco/train2017/000000008520.jpg",
            # "/mnt/petrelfs/zhangyuanhan/Otter/test/58461685608429_.pic.jpg",
        ]

        encoded_frames_list = []
        for url in urls:
            frames = get_image(url)
            encoded_frames_list.append(frames)

        in_context_prompts = []
        in_context_examples = [
            "how can the children in the image benefit from this skiing experience?::The children in the image, along with the rest of the family, can benefit from the skiing experience in several ways. For example, skiing is a physical activity that promotes fitness and overall health since it engages various muscle groups and enhances cardiovascular endurance.",
        ]
        for in_context_input in in_context_examples:
            in_context_prompt, in_context_answer = in_context_input.split("::")
            in_context_prompts.append((in_context_prompt.strip(), in_context_answer.strip()))

        # prompts_input = input("Enter the prompts separated by commas (or type 'quit' to exit): ")
        prompts_input = "how might the children benefit from such a skiing activity?"

        prompts = [prompt.strip() for prompt in prompts_input.split(",")]

        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            response = get_response(encoded_frames_list, prompt, model, image_processor, in_context_prompts)
            print(f"Response: {response}")

        if prompts_input.lower() == "quit":
            break