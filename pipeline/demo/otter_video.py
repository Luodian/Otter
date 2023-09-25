import logging
import mimetypes
import os
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
import sys
import random

sys.path.append("../../src")
# make sure you can properly access the otter folder
from otter_ai import OtterForConditionalGeneration

# Disable warnings
requests.packages.urllib3.disable_warnings()

logging.basicConfig(encoding="utf-8", level=logging.INFO)

random.seed(1113)

# ------------------- Utility Functions -------------------


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image and Video Handling Functions -------------------


def extract_frames(video_path, is_transparent_background, video_fps):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    num_frames = video_fps
    # Otter-video maximum frame
    num_frames = 128 if num_frames >= 128 else num_frames
    frame_step = total_frames // num_frames

    logging.info(
        "total frames: {}, num_frames: {}, frame_step: {}".format(
            total_frames, num_frames, frame_step
        )
    )

    frames = []
    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = video.read()
        if ret:
            if is_transparent_background:
                frame = transparent_background(frame, subtractor)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert("RGB")
            frames.append(frame)

    video.release()
    return frames


def transparent_background(frame, subtractor):
    # get foreground
    fg_mask = subtractor.apply(frame)

    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # mask background
    result = cv2.bitwise_and(frame, frame, mask=thresh)
    return result


def get_image(
    url: str, is_transparent_background: bool, video_fps: float
) -> Union[Image.Image, list]:
    if "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get(
            "Content-Type"
        )

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
        frames = extract_frames(video_path, is_transparent_background, video_fps)
        if "://" in url:  # Only remove the temporary video file if it was downloaded
            os.remove(video_path)
        return frames
    else:
        raise ValueError("Invalid content type. Expected image or video.")


# ------------------- OTTER Prompt and Response Functions -------------------


def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def get_response(
    input_data, prompt: str, model=None, image_processor=None, tensor_dtype=None
) -> str:
    if isinstance(input_data, Image.Image):
        vision_x = (
            image_processor.preprocess([input_data], return_tensors="pt")[
                "pixel_values"
            ]
            .unsqueeze(1)
            .unsqueeze(0)
        )
    elif isinstance(input_data, list):  # list of video frames
        vision_x = (
            image_processor.preprocess(input_data, return_tensors="pt")["pixel_values"]
            .unsqueeze(0)
            .unsqueeze(0)
        )
    else:
        raise ValueError(
            "Invalid input data. Expected PIL Image or list of video frames."
        )

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )

    # Get the data type from model's parameters
    model_dtype = next(model.parameters()).dtype

    # Convert tensors to the model's data type
    vision_x = vision_x.to(dtype=model_dtype)
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    bad_words_id = model.text_tokenizer(
        ["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False
    ).input_ids
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
        bad_words_ids=bad_words_id,
    )
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
