## 🪩 Serving Demo

We will show you how to host a demo on your own computer using gradio.

## Preparation

### Download the checkpoints

The 🦦 Otter checkpoint and the 🦩 Open Flamingo checkpoint can be auto-downloaded with the code below.

## Start Demo 

### Launch a controller

```Shell
python -m pipeline.serve.controller --host 0.0.0.0 --port 10000
```

### Launch a model worker

```Shell
# Init our 🦦 Otter model on GPU
CUDA_VISIBLE_DEVICES=0,1 python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model_name otter --checkpoint_path luodian/otter-9b-hf --num_gpus 2 --limit_model_concurrency 200
# Init our 🦦 Otter video model on CPU
CUDA_VISIBLE_DEVICES=0,1 python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40002 --worker http://localhost:40002 --model_name otter_video --checkpoint_path checkpoint/otter9B_DC_fullset/ --num_gpus 0 --limit_model_concurrency 200
# Init original open flamingo model on GPU
CUDA_VISIBLE_DEVICES=2,3 python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo --checkpoint_path luodian/openflamingo-9b-hf --num_gpus 2 --limit_model_concurrency 200

# Init original open flamingo model on CPU
python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo_original --checkpoint_path luodian/openflamingo-9b-hf --num_gpus 0
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...".

### Launch a gradio web server

```Shell
# Image demo
python -m pipeline.serve.gradio_web_server --controller http://localhost:10000 --port 7861
# Video demo
python -m pipeline.serve.gradio_web_server_video --controller http://localhost:10000 --port 7862
```

Now, you can open your browser and chat with the model!

## Mini Demo

Here is an example of multi-modal ICL (in-context learning) with 🦦 Otter. We provide two demo images with corresponding instructions and answers, then we ask the model to generate an answer given our instruct. You may change your instruction and see how the model responds.

``` python
import requests
import torch
import transformers
from PIL import Image
from otter.modeling_otter import OtterForConditionalGeneration

model = OtterForConditionalGeneration.from_pretrained(
    "luodian/otter-9b-hf", device_map="auto"
)
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)
demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg", stream=True
    ).raw
)
query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True
    ).raw
)
vision_x = (
    image_processor.preprocess(
        [demo_image_one, demo_image_two, query_image], return_tensors="pt"
    )["pixel_values"]
    .unsqueeze(1)
    .unsqueeze(0)
)
model.text_tokenizer.padding_side = "left"
lang_x = model.text_tokenizer(
    [
        "<image>User: what does the image describe? GPT:<answer> two cats sleeping.<|endofchunk|><image>User: what does the image describe? GPT:<answer> a bathroom sink.<|endofchunk|><image>User: what does the image describe? GPT:<answer>"
    ],
    return_tensors="pt",
)
generated_text = model.generate(
    vision_x=vision_x.to(model.device),
    lang_x=lang_x["input_ids"].to(model.device),
    attention_mask=lang_x["attention_mask"].to(model.device),
    max_new_tokens=256,
    num_beams=3,
    no_repeat_ngram_size=3,
)

print("Generated text: ", model.text_tokenizer.decode(generated_text[0]))
```

An example for video.
``` python
``` python
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

from otter.modeling_otter import OtterForConditionalGeneration

# Disable warnings
requests.packages.urllib3.disable_warnings()

# ------------------- Utility Functions -------------------


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image and Video Handling Functions -------------------


def extract_frames(video_path, num_frames=128):
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


def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def get_response(input_data, prompt: str, model=None, image_processor=None) -> str:
    if isinstance(input_data, Image.Image):
        vision_x = (
            image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        )
    elif isinstance(input_data, list):  # list of video frames
        vision_x = image_processor.preprocess(input_data, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )

    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x["input_ids"].to(model.device),
        attention_mask=lang_x["attention_mask"].to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
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


# ------------------- Main Function -------------------

if __name__ == "__main__":
    model = OtterForConditionalGeneration.from_pretrained(
        "checkpoint/otter9B_DC_frame16",
    )
    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    model.eval()

    while True:
        video_url = "dc_demo.mp4"  # Replace with the path to your video file

        frames_list = get_image(video_url)

        prompts_input = input("Enter prompts (comma-separated): ")
        prompts = [prompt.strip() for prompt in prompts_input.split(",")]

        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            response = get_response(frames_list, prompt, model, image_processor)
            print(f"Response: {response}")

        if prompts_input.lower() == "quit":
            break
```