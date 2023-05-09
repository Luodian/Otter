<p align="center" width="100%">
<img src="https://i.postimg.cc/CLPnPvZW/title.png"  width="80%" height="80%">
</p>


<div>
<div align="center">
    <a href='https://brianboli.com/' target='_blank'>Bo Li*</a>&emsp;
    <a href='https://zhangyuanhan-ai.github.io/' target='_blank'>Yuanhan Zhang*</a>&emsp;
    <a href='https://cliangyu.com/' target='_blank'>Liangyu Chen*</a>&emsp;
    <a href='https://king159.github.io/' target='_blank'>Jinghao Wang*</a>&emsp;
    </br>
    <a href='https://jingkang50.github.io/' target='_blank'>Jingkang Yang</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>
</div>
<div>
<div align="center">
    S-Lab, Nanyang Technological University
</div>
 
 -----------------

![](https://img.shields.io/badge/otter-v0.1-darkcyan)
![](https://img.shields.io/github/stars/luodian/otter?style=social)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLuodian%2Fotter&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
![](https://img.shields.io/badge/code%20style-black-000000.svg)

 [Otter-9B (Hugging Face Models)](https://huggingface.co/luodian/otter-9b-hf) | [Youtube Video](https://youtu.be/r-YM4DGGAdE) | [Bilibili Video](https://www.bilibili.com/video/BV1iL411h7HZ/?share_source=copy_web&vd_source=477facaaaa60694f67a784f5eaa905ad) | [Live Demo](https://otter.cliangyu.com/) | [Paper](https://arxiv.org/abs/2305.03726)

## 🦾 Update

- [2023-05-08]
  1. Check our Arxiv release paper at [Otter: A Multi-Modal Model with In-Context Instruction Tuning](https://arxiv.org/abs/2305.03726) !
  2. We support `xformers` for memory efficient attention.

## 🦦 Overview

<div style="text-align:center">
<img src="https://i.postimg.cc/CKgQ2PP7/otter-teaser.png"  width="100%" height="100%">
</div>

Large Language Models (LLMs) have exhibited exceptional universal aptitude as few/zero-shot learners for numerous tasks, thanks to their pre-training on large-scale text data. GPT-3 is a prominent LLM that has showcased significant capabilities in this regard. Furthermore, variants of GPT-3, namely InstrctGPT and ChatGPT, equipped with instruction tuning, have proven effective in interpreting natural language instructions to perform complex real-world tasks. In this paper, we propose to introduce instruction tuning into multi-modal models, motivated by the Flamingo model's upstream interleaved format pretraining dataset. We adopt a similar approach to construct our **MI**-**M**odal **I**n-**C**ontext **I**nstruction **T**uning (**MIMIC-IT**) dataset. We then introduce 🦦 Otter, a multi-modal model based on OpenFlamingo (open-sourced version of DeepMind's Flamingo), trained on MIMIC-IT and showcasing improved instruction-following ability and in-context learning. We also optimize OpenFlamingo's implementation for researchers, democratizing the required training resources from 1x A100 GPU to 4x RTX-3090 GPUs, and integrate both OpenFlamingo and Otter into Hugging Face Transformers for more researchers to incorporate the models into their customized training and inference pipelines.

## 🦦 Examples

<div style="text-align:center">
<img src="https://s2.loli.net/2023/05/08/LIk1SOaKWw73Gz4.png"  width="100%" height="100%">
</div>

---

<div style="text-align:center">
<img src="https://s2.loli.net/2023/05/08/dj8NJeYh9EMtkWb.png"  width="100%" height="100%">
</div>

---

<div style="text-align:center">
<img src="https://s2.loli.net/2023/05/08/rpPX39icK7lVNnF.png"  width="100%" height="100%">
</div>

## 🗂️ Environments

You may install via `conda env create -f environment.yml`. Especially to make sure the `transformers>=4.28.0`, `accelerate>=0.18.0`.

## 🤗 Hugging Face Model

You can use the 🦩 Flamingo model / 🦦 Otter model as a 🤗 Hugging Face model with only a few lines! One-click and then model configs/weights are downloaded automatically.

``` python
from flamingo import FlamingoModel
flamingo_model = FlamingoModel.from_pretrained("luodian/openflamingo-9b-hf", device_map=auto)

from otter import OtterModel
otter_model = OtterModel.from_pretrained("luodian/otter-9b-hf", device_map=auto)
```

Previous [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) was developed with [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) (DDP) on A100 cluster. Loading OpenFlamingo-9B to GPU requires **at least 33G GPU memory**, which is only available on A100 GPUs.

In order to allow more researchers without access to A100 machines to try training OpenFlamingo, we wrap the OpenFlamingo model into a 🤗 hugging Face model ([Jinghao](https://king159.github.io/) has submitted a [PR](https://github.com/huggingface/transformers/pull/23063) to the /huggingface/transformers!). Via `device_map=auto`, the large model is sharded across multiple GPUs when loading and training. This can help researchers who do not have access to A100-80G GPUs to achieve similar throughput in training, testing on 4x RTX-3090-24G GPUs, and model deployment on 2x RTX-3090-24G GPUs. Specific details are below (may vary depending on the CPU and disk performance, as we conducted training on different machines).

<div style="text-align:center">
<img src="https://i.postimg.cc/LsNs55zG/table.png"  width="100%" height="100%">
</div>

---
<div style="text-align:center">
<img src="https://i.postimg.cc/tTcCdcv5/efficiency.png"  width="100%" height="100%">
</div>

Our Otter model is also developed in this way and it's deployed on the 🤗 Hugging Face model hub. Our model can be hosted on two RTX-3090-24G GPUs and achieve a similar speed to one A100-80G machine.



## 🗄 Dataset Preparation

### MultI-Modal In-Context Instruction Tuning (MIMIC-IT) dataset

The OpenFlamingo framework leverages the interleaved multi-modal MMC4 dataset to emerge in its few-shot, in-context learning capabilities. The MMC4 dataset is composed of image-text pairs derived from individual HTML files, with significant contextual relationships between different pairs, as depicted in below Figure (a). An MMC4 training data sample contains (i) a queried image-text pair, where the text typically describes the image, and (ii) context, which includes the remaining image-text pairs from the same HTML file. The primary training objective of OpenFlamingo is to generate text for the queried image-text pair, and the paradigm of generating query text conditioned on in-context examples ensures OpenFlamingo's in-context learning capacity during the inference phase.

Our Multi-Modal In-Context Instruction Tuning (MIMIC-IT) dataset aims to augment OpenFlamingo's instruction comprehension capabilities while preserving its in-context learning capacity. To unleash OpenFlamingo's instruction-following potential, we compile data from visual-language tasks into image-instruction-answer triplets. Concurrently, to maintain OpenFlamingo's in-context learning capacity, we retrieve in-context examples for each triplet, which often lack correlated context, such as a visual question-answer data sample in VQAv2. Specifically, each MIMIC-IT data sample consists of (i) a queried image-instruction-answer triplet, with the instruction-answer tailored to the image, and (ii) context. The context contains a series of image-instruction-answer triplets that contextually correlate with the queried triplet, emulating the relationship between the context and the queried image-text pair found in the MMC4 dataset. The training objective for MIMIC-IT is to generate the answer within the queried image-instruction-answer triplet. The image-instruction-answer triplets are derived from (i) visual question-answer datasets, namely, VQAv2 and GQA, (ii) visual instruction datasets, such as LLaVA, (iii) an in-progress, high-quality panoptic video scene graph dataset from the PVSG repository. For each video, we select 4-8 frames for instruction-following annotation, using the LLaVA dataset as a reference. We have developed three heuristics to construct the context for each image-instruction-answer triplet, as illustrated in Figure (b).

### Example

<div style="text-align:center">
<img src="https://s2.loli.net/2023/05/08/Vyp2eXNtW3xC1Ji.png"  width="100%" height="100%">
</div>

### Preparation

We unify different instructing data into a single dataset [class](pipeline/multi_instruct_data_utils/unify_dataset.py). The full dataset is coming soon! 

## ☄️ Training

Train on `MIMIC-IT` datasets, using the following commands:

First, run, and answer the questions asked. This will generate a config file and save it to the cache folder. The config will be used automatically to properly set the default options when doing `accelerate launch`.

```bash
accelerate config
```

Then run the training script.

```bash
accelerate launch pipeline/train/instruction_following.py \
--pretrained_model_name_or_path=luodian/openflamingo-9b-hf \
--dataset_resampled \
--multi_instruct_path=./in_context_instruct.tsv \
--run_name=otter-9b \
--batch_size=1 \
--num_epochs=6 \
--report_to_wandb \
--cross_attn_every_n_layers=4 \
--lr_scheduler=cosine \
--delete_previous_checkpoint \
--learning_rate=1e-5 \
```

## 💎 Checkpoints

For details, you may refer to the [model card](docs/model_card.md).

## 🪩 Web Demo

We host our [Otter-9B Demo](https://otter.cliangyu.com/) via dual RTX-3090-24G GPUs. Launch your own demo by following the [demo instructions](docs/demo.md).

## 🛠 Incoming Features

We are working towards offering these features to our users. However, we have encountered some issues in the process. If you have the solutions to these issues, we would be grateful if you could submit a pull request with your code. Your contribution would be highly appreciated.

- [x]  `xformers` support: for saving GPU memory and training speedup. issue [#35](https://github.com/Luodian/PET-VLM/issues/35)
- [ ]  `load_in_8bit` support: for saving GPU memory and training speedup. [[issue]()]

### Models

We are working on the following models with much stronger performance.

- [ ] Otter-9B for Videos
- [ ] Otter-15B


## 📑 Citation

If you found this repository useful, please consider citing:
```
@software{li_bo_2023_7879884,
  author       = {Li, Bo and Zhang, Yuanhan and Chen, Liangyu and Wang, Jinghao and Yang, Jingkang and Liu, Ziwei},
  title        = {{Otter: Multi-Modal In-Context Learning Model with Instruction Tuning}},
  month        = apr,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.7879884},
  url          = {https://doi.org/10.5281/zenodo.7879884}
}
```

### 👨‍🏫 Acknowledgements

We thank [Chunyuan Li](https://chunyuan.li/) and [Jack Hessel](https://jmhessel.com/) for their advise and support, as well as the [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) team for their great contribution to the open source community.
