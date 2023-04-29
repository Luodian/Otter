# 🦦 Otter: Multi-Modal In-Context Learning Model with Instruction Tuning
 
## Talk to 🦦 Otter, Give it instruction, and Explore its in-context learning ability!

<!-- [Otter Demo](https://otter.cliangyu.com/) -->

![](https://img.shields.io/badge/otter-chat%20demo-orange?link=http://left&link=https://otter.cliangyu.com)

![](https://img.shields.io/badge/otter-v0.1%20%7C%20alpha-blue)

![](https://img.shields.io/github/stars/luodian/otter?style=social)

## Overview

Recent research emphasizes the importance of instruction tuning in empowering Large Language Models (LLMs), such as boosting GPT-3 to Chat-GPT, to adhere to natural language instruction and effectively accomplish real-world tasks. Flamingo is considered a GPT-3 moment in the multimodal domain. In our project, we propose 🦦 Otter, an in-context instruction-tunning model built upon Flamingo. We enhance its abilities by utilizing a carefully constructed multimodal instruction tuning dataset. Each data sample includes an image-specific instruction along with multiple multimodal instructions, also referred to as multimodal in-context learning examples.

## Environment

You may install via `conda env create -f environment.yml`  or manually install the following packages.

Especially to make sure the `transformers>=4.28.0`, `accelerate==0.19.0.dev0`.

</details>

## Dataset Preparation

Download a subset of the pretraining `multi_instruct` dataset

```bash
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/datasets/pretrain_data/pretrain_data_examples.zip;
unzip pretrain_data_examples.zip ./example_multi_instruct_data
```

## Training

Train on `multi_instruct` example datasets, use following commands:

## 🤗 Hugging Face Model

Right now, you can use the 🦩 Flamingo model as a huggingface model with only two lines!

``` python
from flamingo_hf import FlamingoModel
model = FlamingoModel.from_pretrained("luodian/openflamingo-9b-hf")
```

## Serving Demo

### Launch a controller

```Shell
python -m collie_core.serve.controller --host 0.0.0.0 --port 10000
```

### Launch a model worker

```Shell
export AZURE_DIR="/media/ntu/volume1/home/s121md302_06/data/data/azure"
# Init our model on GPU
CUDA_VISIBLE_DEVICES=0,1 python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model_name otter --checkpoint_path checkpoint/multi_instruct_chunyuan-core_otter9B_lr1e-5_6epochs_hf --num_gpus 2

# Init original model on GPU
CUDA_VISIBLE_DEVICES=2,3 python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo_original --checkpoint_path checkpoint/open_flamingo_9B_hf --num_gpus 2
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...".

### Send a test message

```Shell
python -m collie_core.serve.test_message --model_name LLaVA-13B-v0 --controller http://localhost:10000
```

### Launch a gradio web server

```Shell
python -m collie_core.serve.gradio_web_server --controller http://localhost:10000
```

#### You can open your browser and chat with a model now

## Authors

Equal contribution, alphabetical order.

[Liangyu Chen]()

[Bo Li](https://brianboli.com/)

[Jinghao Wang](https://king159.github.io/)

[Yuanhan Zhang](https://zhangyuanhan-ai.github.io/)

### Acknowledgements

We thank Jingkang Yang and Ziwei Liu for supporting. We thank the Open Flamingo team for their great contribution to the open source community.
