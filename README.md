<header><img src="./assets/collie_icon.png" alt="Collie Icon"><h1>🐾 Collie: A Visual Language Model with Efficient Instruction Tuning</h1></header>

Otter interprets and deciphers complex visual information, enabling seamless integration of images and text. Collie is built on OpenFlamingo.

### Environment

You may install via `conda create -f environment.yaml` or manually install the following packages.
<details>
<summary>Manually Install</summary>

``` bash
conda install pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c conda-forge transformers=4.28.1 -y
conda install -c conda-forge datasets=2.11.0 -y
conda install -c conda-forge wandb=0.14.0 -y
conda install -c conda-forge braceexpand=0.1.5 -y
conda install -c conda-forge webdataset=0.2.48 -y
conda install -c conda-forge scipy=1.10.1 -y
conda install -c conda-forge sentencepiece=0.1.97 -y
conda install -c conda-forge einops=0.6.0 -y
pip install bitsandbytes==0.37.2
conda install -c conda-forge tensorboard=2.12.0 -y
conda install -c conda-forge more-itertools=9.1.0 -y

# install standford-corenlp-full
cd LAVIS/coco-caption;
sh get_stanford_models.sh
```

</details>

### Dataset Preparation

Download a subset pretraining multi_instruct dataset

```bash
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/datasets/pretrain_data/pretrain_data_examples.zip;
unzip pretrain_data_examples.zip ./example_multi_instruct_data
```

### Training

Train on multi_instruct example datasets, use following commands:

### Model Convertion

If you want to train your own model, you need to convert the model to a 🤗 huggingface model format.
Otherwise, you may skip this step and use our 🦦 Otter model directly.

Firstly, download the 🦩 [OpenFlamingo-9B](https://huggingface.co/openflamingo/OpenFlamingo-9B) checkpoint.

Then convert the model to a 🤗 huggingface model format by running the following command

(Notice:

1. You may see some warnings but do not be panic. It is because we only initialize the CLIP-vision encoder from the whole CLIP checkpoint.
2. The entire model will be serialized into 4 files and the total size will be among 32.9G. )

``` bash
python flamingo_hf/converting_flamingo_to_pytorch.py -old 'path/to/flamingo/checkpoint.pt' -new 'folder/to/save/hf_model/'
```

Finally, you may test the converted 🤗 huggingface model using the following python script:
<details>
<summary>test</summary>

``` python
import transformers
from flamingo_hf import (
    FlamingoForConditionalGeneration,
    FlamingoConfig,
)
from PIL import Image
import requests

model = FlamingoForConditionalGeneration.from_pretrained(
"folder/to/save/hf_model/", device_map="auto"
)
tokenizer = model.text_tokenizer
tokenizer.padding_side = (
    "left"  # For generation padding tokens should be on the left
)
lang_x = tokenizer(
    [
        "<image>An image of two cats.<|endofchunk|><image>An image of a baathroom sink.<|endofchunk|><image>An image of"
    ],
    return_tensors="pt",
)
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
image_processor = transformers.CLIPImageProcessor()
vision_x = (
    image_processor.preprocess(
        [demo_image_one, demo_image_two, query_image], return_tensors="pt"
    )["pixel_values"]
    .unsqueeze(1)
    .unsqueeze(0)
)
generated_text = model.generate(
    vision_x=vision_x.to('cuda'),
    lang_x=lang_x["input_ids"].to('cuda'),
    attention_mask=lang_x["attention_mask"].to('cuda'),
    max_new_tokens=20,
    num_beams=3,
)
print("Generated text: ", tokenizer.decode(generated_text[0]))
```

</details>

### Serving Demo

#### Launch a controller

```Shell
python -m collie_core.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a model worker

```Shell
export AZURE_DIR="/media/ntu/volume1/home/s121md302_06/data/data/azure"
# Init our model on GPU
CUDA_VISIBLE_DEVICES=0,1 python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model_name otter --checkpoint_path checkpoint/multi_instruct_chunyuan-core_otter9B_lr1e-5_6epochs_hf --num_gpus 2

# Init original model on GPU
CUDA_VISIBLE_DEVICES=2,3 python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo_original --checkpoint_path checkpoint/open_flamingo_9B_hf --num_gpus 2
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...".

#### Send a test message

```Shell
python -m collie_core.serve.test_message --model_name LLaVA-13B-v0 --controller http://localhost:10000
```

#### Launch a gradio web server

```Shell
python -m collie_core.serve.gradio_web_server --controller http://localhost:10000
```

#### You can open your browser and chat with a model now

### Authors

Equal contribution
