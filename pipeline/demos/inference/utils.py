import mimetypes
import numpy
import yaml
import time
from typing import Union
import requests
from PIL import Image
import sys
from transformers import IdeficsForVisionText2Text, AutoProcessor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.append("/home/luodian/projects/Otter")
from pipeline.train.train_utils import get_image_attention_mask

requests.packages.urllib3.disable_warnings()
import torch


# --- Utility Functions ---
def print_colored(text, color_code):
    end_code = "\033[0m"  # Reset to default color
    print(f"{color_code}{text}{end_code}")


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


def get_image(url: str) -> Union[Image.Image, list]:
    if not url.strip():  # Blank input, return a blank Image
        return Image.new("RGB", (224, 224))  # Assuming 224x224 is the default size for the model. Adjust if needed.
    elif "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    else:
        raise ValueError("Invalid content type. Expected image.")
