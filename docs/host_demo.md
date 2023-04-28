# Demo Hosting

We will show you how to host a demo on your own computer using gradio.

## Preparation

### Download the checkpoints

We assume that you have downloaded the 🦦 Otter checkpoint and the 🦩 Open Flamingo checkpoint.

### Install additional packages

``` bash
pip install fastapi
```

## Launch a controller

``` bash
python -m collie_core.serve.controller --host 0.0.0.0 --port 10000
```

## Launch a model worker

Initialize our 🦦 Otter model on GPU

``` bash

CUDA_VISIBLE_DEVICES=0,1 python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model_name otter --checkpoint_path path/to/the/otter/checkpoint --num_gpus 2

```

Initialize the 🦩 Open Flamingo model on GPU

``` bash
CUDA_VISIBLE_DEVICES=2,3 python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo_original --checkpoint_path path/to/the/flamingo/checkpoint --num_gpus 2
```

Wait until the process finishes loading the model then you will see "Uvicorn running on ...".

## Send a test message

```Shell
python -m collie_core.serve.test_message --model_name LLaVA-13B-v0 --controller http://localhost:10000
```

## Launch a gradio web server

```Shell
python -m collie_core.serve.gradio_web_server --controller http://localhost:10000
```

Now, you can open your browser and chat with the model!

## Small Demo

Here is an example of generating text conditioned on interleaved images/text, in this case we will do few-shot image captioning.

``` python
import requests
import torch
import transformers
from PIL import Image
from otter_hf import OtterForConditionalGeneration
model = OtterForConditionalGeneration.from_pretrained("path/to/the/otter/checkpoint")
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
demo_image_one = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
demo_image_two = Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028137.jpg", stream=True).raw)
query_image = Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True).raw)
vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
model.text_tokenizer.padding_side = "left"
lang_x = model.text_tokenizer(
    ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
    return_tensors="pt",
)
generated_text = model.generate(
    vision_x=vision_x.to(model.device),
    lang_x=lang_x["input_ids"].to(model.device),
    attention_mask=lang_x["attention_mask"].to(model.device),
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", model.text_tokenizer.decode(generated_text[0]))
```
