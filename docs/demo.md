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
CUDA_VISIBLE_DEVICES=0,1 python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model_name otter --checkpoint_path luodian/otter-9b-hf --num_gpus 2

# Init original open flamingo model on GPU
CUDA_VISIBLE_DEVICES=2,3 python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo --checkpoint_path luodian/openflamingo-9b-hf --num_gpus 2 --limit_model_concurrency 200

# Init original open flamingo model on CPU
python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo_original --checkpoint_path luodian/openflamingo-9b-hf --num_gpus 0
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...".

### Launch a gradio web server

```Shell
python -m pipeline.serve.gradio_web_server --controller http://localhost:10000
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
    num_beams=1,
    no_repeat_ngram_size=3,
)

print("Generated text: ", model.text_tokenizer.decode(generated_text[0]))
```
