import argparse
import csv
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROMPT_PATH = "./prompt"


class Config:
    def __init__(
        self,
        folder_name,
        fps,
        fish_variety,
        input_video_path,
    ):
        self.folder_name = folder_name
        self.fps = fps
        self.fish_variety = fish_variety
        self.input_video_path = input_video_path

    @staticmethod
    def from_csv(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} is'nt exist!")
        with open(path) as f:
            reader = csv.reader(f)
            lines = list(map(lambda e: e[0], reader))
        return Config(
            lines[0],
            float(lines[5]),
            int(lines[6]),
            lines[7],
        )


def main():
    # command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv_path",
        type=str,
        required=True,
        help="path for input video metadata csv.",
    )
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default="./output.csv",
        help="path for output csv.",
    )
    parser.add_argument(
        "--truth_csv_path",
        type=str,
        help="path for ground truth csv of output csv. if set calculate accuracy.",
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

    config = Config.from_csv(args.input_csv_path)

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
    video_url = os.path.join(config.folder_name, config.input_video_path)
    if video_url[-3:] != "MP4" and video_url[-3:] != "mp4":
        logger.error("Please set path for mp4.")
        return
    if not os.path.isfile(video_url):
        raise FileNotFoundError(
            f"{video_url} is'nt exist! Change the first line of the input CSV."
        )

    frames_list = get_image(video_url, args.is_transparent_background, int(config.fps))

    # read prompt from ./prompt/*.txt
    result = {}
    prompt_files = glob.glob(os.path.join(PROMPT_PATH, "*.txt"))
    for prompt_file in prompt_files:
        fish_name = os.path.splitext(os.path.basename(prompt_file))[0]
        with open(prompt_file, "r", encoding="utf-8") as file:
            prompt = file.read()
            logger.info(f"Prompt: {prompt}")
            # otter model input and output
            logger.info("Generating answer...")
            response = get_response(
                frames_list, prompt, model, image_processor, tensor_dtype
            )
            logger.info(f"Response: {response}")
            # extract fish number
            fish_num = utils.extract_number_from_response(response)
            result[fish_name] = fish_num
    logger.info(f"Result: {result}")

    result = utils.convert_fish_name_to_number(result)
    logger.info(f"Converted Result: {result}")

    # calculate accuracy
    if args.truth_csv_path:
        ground_truth = {}
        with open(args.truth_csv_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if reader.line_num < 3:
                    continue
                ground_truth[int(row[0])] = int(row[1])
        accuracy = utils.accuracy(result, ground_truth, config.fish_variety)
        logger.info(f"Accuracy: {accuracy}")

    # output
    utils.output_csv(result, video_url, args.output_csv_path, config.fish_variety)


if __name__ == "__main__":
    main()
