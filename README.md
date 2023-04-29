<p align="center" width="100%">
<img src="assets/title.png"  width="80%" height="80%">
</p>


<div>
<div align="center">
    <a href='https://brianboli.com/' target='_blank'>Bo Li*</a>&emsp;
    <a href='https://zhangyuanhan-ai.github.io/' target='_blank'>Yuanhan Zhang*</a>&emsp;
    <a href='https://king159.github.io/' target='_blank'>Jinghao Wang*</a>&emsp;
    <a href='https://cliangyu.com/' target='_blank'>Liangyu Chen*</a>&emsp;
    </br>
    <a href='https://jingkang50.github.io/' target='_blank'>Jingkang Yang</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>
</div>
<div>
<div align="center">
    S-Lab, Nanyang Technological University
</div>
 
 -----------------

![](https://img.shields.io/badge/otter-v0.1-orange)
![](https://img.shields.io/github/stars/luodian/otter?style=social)

[Otter Demo](https://otter.cliangyu.com/) | [Otter-9B (Huggingface Models)](https://huggingface.co/luodian/otter-9b-hf) | [Paper]() | [Video](https://otter.cliangyu.com/)


## 🦦 Overview

Recent research emphasizes the importance of instruction tuning in empowering Large Language Models (LLMs), such as boosting GPT-3 to Chat-GPT, to adhere to natural language instruction and effectively accomplish real-world tasks. Flamingo is considered a GPT-3 moment in the multimodal domain. In our project, we propose 🦦 Otter, an in-context instruction-tuned model built upon Flamingo. We enhance its chat abilities by utilizing a carefully constructed multimodal instruction tuning dataset. Each data sample includes an image-specific instruction along with multiple multimodal instructions, also referred to as multimodal in-context learning examples.

## 🗂️ Environment

# install standford-corenlp-full
cd LAVIS/coco-caption;
sh get_stanford_models.sh
```
</details>
<!-- # Highlight
Recent studies emphasize the importance of instructions for Large Language Models (LLMs), like GPT-3, in completing real-world tasks. Flamingo, a GPT-3 moment in the multimodal domain, excels in multimodal in-context learning, showcasing its ability to follow multimodal instructions, a.k.a. multimodal in-context examples. We aims to enhance Flamingo's multimodal capabilities using a carefully curated instruction following dataset. We present Otter, which can tackle diverse multimodal tasks, ranging from detailed descriptions to complex reasoning, by being guided through (1) an image, (2) an image-specific instruction, and (3) multiple multimodal instructions (multimodal in-context learning examples.)

<!-- # Why we need instruction tuning, and why we choose Flamingo?
- Recent research emphasizes the importance of instruction tuning in empowering Large Language Models (LLMs), such as GPT-3, to adhere to natural language instruction and effectively accomplish real-world tasks. This procedure is essential for improving the zero-and few-shot generalization abilities of LLMs, which are trained using noisy web data.

- Flamingo is considered a GPT-3 moment in the multimodal domain, due to its remarkable performance in visual in-context learning. As visual in-context examples provide multimodal instructions for visual-language models, Flamingo's proficiency in visual in-context learning indicates its capability to follow multimodal instructions.

- In our project, we aim to enhance Flamingo's multimodal instruction-following abilities by utilizing a carefully constructed multimodal instruction tuning dataset. Each data sample includes an image-specific instruction along with multiple multimodal instructions, also referred to as multimodal in-context learning examples. --> 

## Multi-model instruction tuning dataset with in-context examples
The pre-training process for the Open-Flamingo model employs the MMC4 interleaved multimodality dataset to endow the model with in-context few-shot learning capabilities. The development of our instruction-following dataset adheres to the guiding principles of MMC4, which dictate that the instruction and image examples incorporated into the context should exhibit semantic pertinence to the query instruction and image.

1. To augment the LLAVA dataset, we retrieve in-context examples for each query data.
2. We curate high-quality video data from the Video PSG repository (https://github.com/Jingkang50/OpenPSG). For each video, we select 4-8 frames to be annotated for instruction-following, using the LLAVA dataset as a reference. During the training phase, given a frame, we opt for additional frames, along with their corresponding instructions and answers, to serve as in-context examples.

### Details
<img src="./images/image_example_4.png" alt="Description" width="1200" height="200"> 
For details of our training data,  check our [dataset card](/docs/dataset_card.md).


### Preparation

You can use the 🦩 Flamingo model / 🦦 Otter model as a huggingface model with only a few lines! One-click and then model configs/weights are downloaded automatically.

``` python
from flamingo import FlamingoModel
flamingo_model = FlamingoModel.from_pretrained("luodian/openflamingo-9b-hf")

from otter import OtterModel
otter_model = OtterModel.from_pretrained("luodian/otter-9b-hf")
```

## 🗄 Dataset Preparation

### Multi-model instruction tuning dataset with in-context examples
The pre-training process for the Open-Flamingo model employs the MMC4 interleaved multimodality dataset to endow the model with in-context few-shot learning capabilities. The development of our instruction-following dataset adheres to the guiding principles of MMC4, which dictate that the instruction and image examples incorporated into the context should exhibit semantic pertinence to the query instruction and image.

1. To augment the LLAVA dataset, we retrieve in-context examples for each query data.
2. We curate high-quality video data from the Video PSG repository (https://github.com/Jingkang50/OpenPSG). For each video, we select 4-8 frames to be annotated for instruction-following, using the LLAVA dataset as a reference. During the training phase, given a frame, we opt for additional frames, along with their corresponding instructions and answers, to serve as in-context examples.


### Example

<p align="center" width="100%"><img src="assets/image_example_3.png" alt="otter-example" style="width: 100%; min-width: 300px; display: block; margin: auto;"></a></p>


### Preparation

We unify different instructing data into a single dataset [class](pipeline/multi_instruct_data_utils/unify_dataset.py). Full dataset is comming soon! 


## ☄️ Training

Train on `multi_instruct` example datasets, use following commands:

For details, you may refer to the [model card](docs/model_card.md).

## 🪩 Web Demo
We host our [Otter Demo](https://otter.cliangyu.com/) via dual RTX-3090. Launch your own demo by following [instructions](docs/demo.md).

### 🛠 Incoming Support

### Features

We are working on the following features. We are working hard to provide these features. Here are some of the issues we have encountered. If you know the answers, please feel free to submit a pull request with your code. We will be very grateful.

- [ ]  `xformers` support: for saving GPU memory and training speedup. [[issue]()]
- [ ]  `load_in_8bit` support: for saving GPU memory and training speedup. [[issue]()]

### Models

We are working on the following models with much stronger performance.

- [ ] Otter-9B for Videos
- [ ] Otter-15B

### 👨‍🏫 Acknowledgements 

We thank [Chunyuan Li](https://chunyuan.li/) and [Jack Hessel](https://jmhessel.com/) for their advise and support, as well as the Open Flamingo team for their great contribution to the open source community.
