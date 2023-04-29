<p align="center" width="100%">
<img src="assets/title.png"  width="80%" height="80%">
</p>
 

![](https://img.shields.io/badge/code-v0.1%20%7C%20alpha-blue)
![](https://img.shields.io/badge/demo-otter%20chat-orange?link=http://left&link=https://otter.cliangyu.com)
![](https://img.shields.io/github/stars/luodian/otter?style=social)

## 🦦 Overview

Recent research emphasizes the importance of instruction tuning in empowering Large Language Models (LLMs), such as boosting GPT-3 to Chat-GPT, to adhere to natural language instruction and effectively accomplish real-world tasks. Flamingo is considered a GPT-3 moment in the multimodal domain. In our project, we propose 🦦 Otter, an in-context instruction-tuned model built upon Flamingo. We enhance its chat abilities by utilizing a carefully constructed multimodal instruction tuning dataset. Each data sample includes an image-specific instruction along with multiple multimodal instructions, also referred to as multimodal in-context learning examples.

## 🗂️ Environment

You may install via `conda env create -f environment.yml`. Especially to make sure the `transformers>=4.28.0`, `accelerate==0.19.0.dev0`.

## 🤗 Hugging Face Model

Previous OpenFlamingo was developed with DDP and it's not easy to implement a fully sharded mechanism. Loading OpenFlamingo-9B to GPU memory requires >33G GPU memory.

To accelerate and demoncratize it, we wrap the Open Flamingo model into a 🤗 huggingface model (and submit a [PR](https://github.com/huggingface/transformers/pull/23063) to the /huggingface/transformers!). We use `accelerator` to speed up our training and implement it in a fully sharded mechanism across multiple GPUs. 

This can help researchers who do not have access to A100-80G GPUs to achieve the same throughput in training, testing on 4x3090-24G GPUs, and model deployment on 2x3090-24G GPUs. Specific details are below.

<div style="text-align:center">
<img src="assets/table.png"  width="100%" height="100%">
</div>

<div style="text-align:center">
<img src="assets/efficiency.png"  width="100%" height="100%">
</div>

Our Otter model is also developed in this way and it's deployed on the 🤗 Hugging Face model hub.

You can use the 🦩 Flamingo model / 🦦 Otter model as a huggingface model with only a few lines! One-click and then model configs/weights are downloaded automatically.

``` python
from flamingo import FlamingoModel
flamingo_model = FlamingoModel.from_pretrained("luodian/openflamingo-9b-hf")
# install standford-corenlp-full
cd LAVIS/coco-caption;
sh get_stanford_models.sh
```
<!-- # Highlight
Recent studies emphasize the importance of instructions for Large Language Models (LLMs), like GPT-3, in completing real-world tasks. Flamingo, a GPT-3 moment in the multimodal domain, excels in multimodal in-context learning, showcasing its ability to follow multimodal instructions, a.k.a. multimodal in-context examples. We aims to enhance Flamingo's multimodal capabilities using a carefully curated instruction following dataset. We present Otter, which can tackle diverse multimodal tasks, ranging from detailed descriptions to complex reasoning, by being guided through (1) an image, (2) an image-specific instruction, and (3) multiple multimodal instructions (multimodal in-context learning examples.)

<<<<<<< HEAD
<!-- # Highlight
Recent studies emphasize the importance of instructions for Large Language Models (LLMs), like GPT-3, in completing real-world tasks. Flamingo, a GPT-3 moment in the multimodal domain, excels in multimodal in-context learning, showcasing its ability to follow multimodal instructions, a.k.a. multimodal in-context examples. We aims to enhance Flamingo's multimodal capabilities using a carefully curated instruction following dataset. We present Otter, which can tackle diverse multimodal tasks, ranging from detailed descriptions to complex reasoning, by being guided through (1) an image, (2) an image-specific instruction, and (3) multiple multimodal instructions (multimodal in-context learning examples.)

### Multi-model instruction tuning dataset with in-context examples
The pre-training process for the Open-Flamingo model employs the MMC4 interleaved multimodality dataset to endow the model with in-context few-shot learning capabilities. The development of our instruction-following dataset adheres to the guiding principles of MMC4, which dictate that the instruction and image examples incorporated into the context should exhibit semantic pertinence to the query instruction and image.

1. To augment the LLAVA dataset, we retrieve in-context examples for each query data.
2. We curate high-quality video data from the Video PSG repository (https://github.com/Jingkang50/OpenPSG). For each video, we select 4-8 frames to be annotated for instruction-following, using the LLAVA dataset as a reference. During the training phase, given a frame, we opt for additional frames, along with their corresponding instructions and answers, to serve as in-context examples.

For details, you may refer to the [dataset card](docs/dataset_card.md).

## ☄️ Training

## 🗄 Dataset Preparation

### Multi-model instruction tuning dataset with in-context examples
The pre-training process for the Open-Flamingo model employs the MMC4 interleaved multimodality dataset to endow the model with in-context few-shot learning capabilities. The development of our instruction-following dataset adheres to the guiding principles of MMC4, which dictate that the instruction and image examples incorporated into the context should exhibit semantic pertinence to the query instruction and image.

1. To augment the LLAVA dataset, we retrieve in-context examples for each query data.
2. We curate high-quality video data from the Video PSG repository (https://github.com/Jingkang50/OpenPSG). For each video, we select 4-8 frames to be annotated for instruction-following, using the LLAVA dataset as a reference. During the training phase, given a frame, we opt for additional frames, along with their corresponding instructions and answers, to serve as in-context examples.

### Examples
<img src="./images/image_example_4.png" alt="Description" width="1200" height="200"> 


## ☄️ Training

Train on `multi_instruct` example datasets, use following commands:

For details, you may refer to the [model card](docs/model_card.md).

## 🪩 Web Demo
We host our [Otter Demo](https://otter.cliangyu.com/) via dual RTX-3090. Launch your own demo by following [instructions](docs/demo.md).

## 🛠 Incoming Features

We are working on the following features. We are working hard to provide these features. Here are some of the issues we have encountered. If you know the answers, please feel free to submit a pull request with your code. We will be very grateful.

- `xformers` support: for saving GPU memory and training speedup. [[issue](https://github.com/Luodian/PET-VLM/issues/35)]
- `load_in_8bit` support: for saving GPU memory and training speedup. [[issue]()]


## 👨‍💻 Authors

Equal contribution, alphabetical order.

[Liangyu Chen](https://cliangyu.com/)

[Bo Li](https://brianboli.com/)

[Jinghao Wang](https://king159.github.io/)

[Yuanhan Zhang](https://zhangyuanhan-ai.github.io/)

### 👨‍🏫 Acknowledgements 

We thank [Chunyuan Li](https://chunyuan.li/) and [Jack Hessel](https://jmhessel.com/) for their advise and support, as well as the Open Flamingo team for their great contribution to the open source community.
