import argparse
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
from src.otter_ai import OtterForConditionalGeneration
from pipeline.demo.otter_video import get_response, get_image

requests.packages.urllib3.disable_warnings()

parser = argparse.ArgumentParser()
parser.add_argument("--input_video_path", type=str, required=True, help="path for input mp4 video")
args = parser.parse_args()

load_bit = "fp32"
if load_bit == "fp16":
    precision = {"torch_dtype": torch.float16}
elif load_bit == "bf16":
    precision = {"torch_dtype": torch.bfloat16}
elif load_bit == "fp32":
    precision = {"torch_dtype": torch.float32}

model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-9B-DenseCaption", device_map="auto", **precision)
tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[load_bit]

model.text_tokenizer.padding_side = "left"
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
model.eval()

video_url = args.input_video_path

frames_list = get_image(video_url)

# TODO: プロンプトのファイル数だけ繰り返す
prompts_input = input("Enter prompts: ")
response = get_response(frames_list, prompts_input, model, image_processor, tensor_dtype)
print(f"Response: {response}")

# TODO: 回答をCSVに書き込む
