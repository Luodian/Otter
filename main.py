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
import logging
import glob
import utils

sys.path.append("./src")
from src.otter_ai import OtterForConditionalGeneration
from pipeline.demo.otter_video import get_response, get_image

requests.packages.urllib3.disable_warnings()

logger = logging.getLogger("alcon_log")
logger.setLevel(logging.INFO)

PROMPT_PATH = "./prompt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video_path", type=str, required=True, help="path for input mp4 video."
    )
    parser.add_argument(
        "--output_csv_path", type=str, default="output.csv", help="path for output csv."
    )
    parser.add_argument(
        "--is_transparent_background",
        type=bool,
        default=True,
        help="Transparent background or not.",
    )
    parser.add_argument(
        "--load_bit",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="model load bit",
    )
    args = parser.parse_args()

    # model settings
    load_bit = args.load_bit
    if load_bit == "fp16":
        precision = {"torch_dtype": torch.float16}
    elif load_bit == "bf16":
        precision = {"torch_dtype": torch.bfloat16}
    elif load_bit == "fp32":
        precision = {"torch_dtype": torch.float32}

    model = OtterForConditionalGeneration.from_pretrained(
        "luodian/OTTER-9B-DenseCaption", device_map="auto", **precision
    )
    tensor_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[load_bit]

    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    model.eval()

    # read mp4
    video_url = args.input_video_path

    if video_url[-3:] != "MP4" and video_url[-3:] != "mp4":
        logger.error("mp4ファイルを指定してください。")
        return

    frames_list = get_image(video_url, args.is_transparent_background)

    # TODO: プロンプトのファイル数だけ繰り返す
    prompt_files = glob.glob(os.path.join(PROMPT_PATH, "*.txt"))
    print(prompt_files)
    for prompt_file in prompt_files:
        with open(prompt_file, "r") as file:
            prompt = file.read()
            # otter model input and output
            logger.info("Generating answer...")
            response = get_response(
                frames_list, prompt, model, image_processor, tensor_dtype
            )
            fish_num = utils.extract_number_from_response(response)
            # TODO: CSVに書き込む
            logger.info(f"Response: {response}")


if __name__ == "__main__":
    main()
