<p align="center" width="100%">
<img src="https://i.postimg.cc/MKmyP9wH/new-banner.png"  width="80%" height="80%">
</p>


<div>
<div align="center">
    <a href='https://brianboli.com/' target='_blank'>Bo Li*<sup>,1</sup></a>&emsp;
    <a href='https://zhangyuanhan-ai.github.io/' target='_blank'>Yuanhan Zhang*<sup>,1</sup></a>&emsp;
    <a href='https://cliangyu.com/' target='_blank'>Liangyu Chen*<sup>,1</sup></a>&emsp;
    <a href='https://king159.github.io/' target='_blank'>Jinghao Wang*<sup>,1</sup></a>&emsp;
    <a href='https://pufanyi.github.io/' target='_blank'>Fanyi Pu*<sup>,1</sup></a>&emsp;
    </br>
    <a href='https://jingkang50.github.io/' target='_blank'>Jingkang Yang<sup>1</sup></a>&emsp;
    <a href='https://chunyuan.li/' target='_blank'>Chunyuan Li<sup>2</sup></a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>1</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp;
    <sup>2</sup>Microsoft Research, Redmond
</div>
 
 -----------------

![](https://img.shields.io/badge/otter-v0.2-darkcyan)
![](https://img.shields.io/github/stars/luodian/otter?style=social)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLuodian%2Fotter&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
![](https://img.shields.io/badge/code%20style-black-000000.svg)

[Project Page](https://otter-ntu.github.io/) | [Otter Paper](https://arxiv.org/abs/2305.03726) | [MIMIC-IT Paper](https://arxiv.org/abs/2306.05425) | [MIMIC-IT Dataset](mimic-it/readme.md)

**Video Demo:** [Otter's Conceptual Demo Video](https://www.youtube.com/watch?v=K8o_LKGQJhs) | [Bilibili](https://www.bilibili.com/video/BV1Bo4y1T7SN/?share_source=copy_web&vd_source=477facaaaa60694f67a784f5eaa905ad)

**Interactive Demo:** [Otter Demo (video version, upcoming)]()

**Checkpoints:** [Checkpoints v0.1](https://huggingface.co/luodian/otter-9b-hf) | [Checkpoints v0.2 upcoming)]()

Otter v0.2 supports videos inputs (frames are arranged as original Flamingo's implementation) and multiple images inputs (they serve as in-context examples for each other). 

**Eval Results:** [Multi-Modal Arena](http://vlarena.opengvlab.com/) | [Multi-Modal AGI Benchmark (Upcoming)]()

<!-- [Youtube Video](https://www.youtube.com/watch?v=K8o_LKGQJhs) | [Bilibili Video](https://www.bilibili.com/video/BV1Bo4y1T7SN/?share_source=copy_web&vd_source=477facaaaa60694f67a784f5eaa905ad) | 📝[Paper]() -->

<div style="text-align:center">
<img src="https://i.postimg.cc/Tw1Z0BCW/otterv0-2-demo.png"  width="100%" height="100%">
</div>

## 🦾 Update

- [2023-06-08]
1. Introducing Project Otter's brand new homepage: https://otter-ntu.github.io/. Check it out now!
2. Check our [paper](https://arxiv.org/abs/2306.05425) introducing MIMIC-IT in details. Meet MIMIC-IT, the first multimodal in-context instruction tuning dataset with 2.8M instructions! Designed to create diverse vision-language instructions that align with real-world visual content, MIMIC-IT spans across seven image and video datasets covering a vast array of scenes. From general scene understanding to spotting subtle differences and enhancing egocentric view comprehension for AR headsets, our MIMIC-IT dataset has it all. Discover more about the MIMIC-IT dataset now!
3. Stay tuned for our upcoming Otter Model v0.2, trained on the MIMIC-IT dataset! With the ability to understand daily scenes, reason in context, spot differences in observations, and act as an egocentric assistant. Checkout conceptual demo video at [Youtube](https://www.youtube.com/watch?v=K8o_LKGQJhs) or [Bilibili](https://www.bilibili.com/video/BV1Bo4y1T7SN/?share_source=copy_web&vd_source=477facaaaa60694f67a784f5eaa905ad)!

- [2023-05-14]
  1. Otter battles with Owl? the Pokémon Arena is here! Our model is selected into [Multi-Modal Arena](http://vlarena.opengvlab.com/). This is an interesting Multi-Modal Foundation Models competition arena that let you see different models reaction to the same question.

- [2023-05-08]
  1. Check our Arxiv release paper at [Otter: A Multi-Modal Model with In-Context Instruction Tuning](https://arxiv.org/abs/2305.03726) !
  2. We support `xformers` for memory efficient attention.

## 🦦 Overview

Large Language Models (LLMs) have exhibited exceptional universal aptitude as few/zero-shot learners for numerous tasks, thanks to their pre-training on large-scale text data. GPT-3 is a prominent LLM that has showcased significant capabilities in this regard. Furthermore, variants of GPT-3, namely InstrctGPT and ChatGPT, equipped with instruction tuning, have proven effective in interpreting natural language instructions to perform complex real-world tasks. In this paper, we propose to introduce instruction tuning into multi-modal models, motivated by the Flamingo model's upstream interleaved format pretraining dataset. We adopt a similar approach to construct our **MI**-**M**odal **I**n-**C**ontext **I**nstruction **T**uning (**MIMIC-IT**) dataset. We then introduce 🦦 Otter, a multi-modal model based on OpenFlamingo (open-sourced version of DeepMind's Flamingo), trained on MIMIC-IT and showcasing improved instruction-following ability and in-context learning. We integrate both OpenFlamingo and Otter into Hugging Face Transformers for more researchers to incorporate the models into their customized training and inference pipelines.

## 🗄 MIMIC-IT Dataset Details

<p align="center" width="100%">
<img src="https://i.postimg.cc/yYMm1G5X/mimicit-logo.png"  width="80%" height="80%">
</p>

MIMIC-IT covers a vast array of real-life scenarios that empower Vision-Language Models (VLMs) to not only comprehend general scenes, but also to reason about context and astutely differentiate between observations. MIMIC-IT also enables the application of egocentric visual assistant model that can serve that can answer your questions like **Hey, Do you think I left my keys on the table?**. In addition to English, MIMIC-IT is also multilingual, supporting Chinese, Korean, Japanese, German, French, Spanish, and Arabic, thereby allowing a larger global audience to altogether enjoy from the convenience brought about by advancements in artificial intelligence.

<p align="center" width="100%">
<img src="https://i.postimg.cc/RCGp0vQ1/syphus.png"  width="80%" height="80%">
</p>

We also introduce **Syphus**, an automated pipeline for generating high-quality instruction-response pairs in multiple languages. Building upon the framework proposed by LLaVA, we utilize ChatGPT to generate instruction-response pairs based on visual content. To ensure the quality of the generated instruction-response pairs, our pipeline incorporates system messages, visual annotations, and in-context examples as prompts for ChatGPT. 

<!-- System messages define the desired tone and style of the generated instruction-response pairs, while visual annotations provide essential image information such as bounding boxes and image descriptions. In-context examples assist ChatGPT in learning within the context.  During cold-start stage, in-context examples are collected by prompting ChatGPT solely through system messages and visual annotations, employing a heuristic approach. This stage concludes only when a satisfactory in-context examples are identified. In step 4, once the instruction-response pairs are obtained, the pipeline expands them into Chinese (zh), Japanese (ja), Spanish (es), German (de), French (fr), Korean (ko), and Arabic (ar). -->

For more details, please check the [MIMIC-IT dataset](mimic-it/README.md).


## 🤖 Otter Model Details

<div style="text-align:center">
<img src="https://i.postimg.cc/CKgQ2PP7/otter-teaser.png"  width="100%" height="100%">
</div>

Otter is designed to support multi-modal in-context instruction tuning based on the OpenFlamingo model, which involves conditioning the language model on the corresponding media, such as an image that corresponds to a caption or an instruction-response pair. 

We train Otter on MIMIC-IT dataset with approximately 2.8 million in-context instruction-response pairs, which are structured into a cohesive template to facilitate various tasks.

The following template encompasses images, user instructions, and model-generated responses, utilizing the `Human` and `Assistant` role labels to enable seamless user-assistant interactions.

```
<image>Human:{instruction} Assistant:<answer>{response}<endofchunk>
```

Training the Otter model on the MIMIC-IT dataset allows it to acquire different capacities, as demonstrated by the LA and SD tasks. Trained on the LA task, the model exhibits exceptional scene comprehension, reasoning abilities, and multi-round conversation capabilities. 

```python
<image>Human:{instruction} Assistant:<answer>{response}<endofchunk>
```

Regarding the concept of organizing visual-language in-context examples, we demonstrate here the acquired ability of the Otter model to follow inter-contextual instructions after training on the LA-T2T task. The organized input data format is as follows:

```python
# Multiple in-context example with similar instructions
<image>Human:{instruction} Assistant:<answer>{response}<|endofchunk|>
# ....
<image>Human:{instruction} Assistant:<answer>{response}<|endofchunk|>
# Query example
<image>Human:{instruction} Assistant:<answer>
```

For more details, please refer to our [paper](https://arxiv.org/abs/2306.05425)'s appendix for other tasks.
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

<!-- ---
<div style="text-align:center">
<img src="https://i.postimg.cc/tTcCdcv5/efficiency.png"  width="100%" height="100%">
</div> -->

Our Otter model is also developed in this way and it's deployed on the 🤗 Hugging Face model hub. Our model can be hosted on two RTX-3090-24G GPUs and achieve a similar speed to one A100-80G machine.

## ☄️ Training

Train on `MIMIC-IT` datasets, using the following commands:

First, run, and answer the questions asked. This will generate a config file and save it to the cache folder. The config will be used automatically to properly set the default options when doing `accelerate launch`.

```bash
accelerate config
```

Then run the training script.

```bash
accelerate launch --config_file=./accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path=path/to/otter_9b_hf  \
--dataset_resampled \
--multi_instruct_path="path/to/instruction.json" \
--images_path="path/to/image.json" \
--train_config_path="/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/DC/DC_train.json" \
--batch_size=4 \
--num_epochs=3 \
--report_to_wandb \
--wandb_entity=ntu-slab \
--run_name=otter9B_DC_frame16 \
--wandb_project=otter9B \
--workers=1 \
--cross_attn_every_n_layers=4 \
--lr_scheduler=cosine \
--delete_previous_checkpoint \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
```

## 💎 Checkpoints

For details, you may refer to the [model card](docs/model_card.md).

## 🪩 Web Demo

We host our [Otter-9B Demo](https://otter.cliangyu.com/) via dual RTX-3090-24G GPUs. Launch your own demo by following the [demo instructions](docs/demo.md).

## 🛠 Incoming Features

We are working towards offering these features to our users. However, we have encountered some issues in the process. If you have the solutions to these issues, we would be grateful if you could submit a pull request with your code. Your contribution would be highly appreciated.

- [x]  `xformers` support: for saving GPU memory and training speedup. issue [#35](https://github.com/Luodian/PET-VLM/issues/35)
- [ ]  `load_in_8bit` support: for saving GPU memory and training speedup. [[issue]()]

## 📑 Citation

If you found this repository useful, please consider citing:
```
@article{li2023otter,
  title={Otter: A Multi-Modal Model with In-Context Instruction Tuning},
  author={Li, Bo and Zhang, Yuanhan and Chen, Liangyu and Wang, Jinghao and Yang, Jingkang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2305.03726},
  year={2023}
}

@article{li2023mimicit,
  title={Otter: A Multi-Modal Model with In-Context Instruction Tuning},
  author={Li, Bo and Zhang, Yuanhan and Chen, Liangyu and Wang, Jinghao and Yang, Jingkang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2305.03726},
  year={2023}
}

@article{li2023mimicit,
    title={MIMIC-IT: Multi-Modal In-Context Instruction Tuning},
    author={Bo Li and Yuanhan Zhang and Liangyu Chen and Jinghao Wang and Fanyi Pu and Jingkang Yang and Chunyuan Li and Ziwei Liu},
    year={2023},
    eprint={2306.05425},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

```

### 👨‍🏫 Acknowledgements

We thank [Chunyuan Li](https://chunyuan.li/) and [Jack Hessel](https://jmhessel.com/) for their advise and support, as well as the [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) team for their great contribution to the open source community.
