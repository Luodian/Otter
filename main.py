import mimetypes
import os
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
import sys

sys.path.append("./src")
# make sure you can properly access the otter folder
from src.otter_ai import OtterForConditionalGeneration
from pipeline.demo.otter_video import get_response, get_image

# from Otter.pipeline.demo

# Disable warnings
requests.packages.urllib3.disable_warnings()


# ------------------- Main Function -------------------
load_bit = "fp32"
if load_bit == "fp16":
    precision = {"torch_dtype": torch.float16}
elif load_bit == "bf16":
    precision = {"torch_dtype": torch.bfloat16}
elif load_bit == "fp32":
    precision = {"torch_dtype": torch.float32}

# This model version is trained on MIMIC-IT DC dataset.
model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-9B-DenseCaption", device_map="auto", **precision)
tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[load_bit]

model.text_tokenizer.padding_side = "left"
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
model.eval()

while True:
    video_url = input("Enter video path: ")  # Replace with the path to your video file, could be any common format.

    frames_list = get_image(video_url)

    while True:
        prompts_input = input("Enter prompts: ")

        if prompts_input.lower() == "quit":
            break

        print(f"\nPrompt: {prompts_input}")
        response = get_response(frames_list, prompts_input, model, image_processor, tensor_dtype)
        print(f"Response: {response}")
