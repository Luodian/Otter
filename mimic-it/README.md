<p align="center" width="100%">
<img src="./docs/mimicit_logo.png"  width="80%" height="80%">
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

<!-- ![](https://img.shields.io/badge/otter-v0.1-darkcyan)
![](https://img.shields.io/github/stars/luodian/otter?style=social)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLuodian%2Fotter&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
![](https://img.shields.io/badge/code%20style-black-000000.svg) -->

[Project Page](https://otter-ntu.github.io/) | [Youtube Video](https://www.youtube.com/watch?v=K8o_LKGQJhs) | [Bilibili Video](https://www.bilibili.com/video/BV1Bo4y1T7SN/?share_source=copy_web&vd_source=477facaaaa60694f67a784f5eaa905ad) | 📝[Paper]()

## 🌳 MIMIC-IT Overview

MIMIC-IT covers a vast array of real-life scenarios that empower Vision-Language Models (VLMs) to not only comprehend general scenes, but also to reason about context and astutely differentiate between observations. MIMIC-IT also enables the application of egocentric visual assistant model that can serve that can answer your questions like **Hey, Do you think I left my keys on the table?**. In addition to English, MIMIC-IT is also multilingual, supporting Chinese, Korean, Japanese, German, French, Spanish, and Arabic, thereby allowing a larger global audience to altogether enjoy from the convenience brought about by advancements in artificial intelligence.

High-quality instructions are essential for the zero-shot performance of large language models on interactive natural language tasks. For interactive vision-language tasks involving intricate visual scenes, a large quantity of diverse and creative instructions should be imperative to tune vision-language models (VLMs). Nevertheless, the current availability of vision-language instructions in terms of quantity, diversity, and creativity remains limited, posing challenges to the generalization of interactive VLMs. Here we present **MIMIC-IT**, a dataset comprising 2.8M multi-modal instructions-response pairs based on images and videos. Each instruction-response pair is accompanied by multi-modal in-context information, forming conversational contexts aimed at empowering VLMs in perception, reasoning, and planning. The instruction-response collection process, dubbed as $Syphus$, is scaled using an automatic annotation pipeline that combines human expertise with GPT's capabilities.

<p align="center" width="100%">
<img src="https://i.postimg.cc/k406BN26/mimic-it.png"  width="80%" height="80%">
</p>
## Dataset Statistics

<!-- <img src="docs/teaser.pdf"  width="80%" height="80%"> -->

Dataset | # Image |# Instruction| Abbr.
-- | -- | -- | --
LLAVA-Complex Reasoning | 81,398 | 76,643 | LACR
LLAVA-Detailed Description | 81,398 | 23,240 | LADD
LLAVA-Conversation | 81,398 | 56,681 | LACONV
SpotTheDifference | 9,524 | 15,989 | SD
SceneNavigation | 562 | 6,640 | SN
DenseCaption | 10,009 | 62,536 | DC
TVCaption | 86,603 | 92,828 | TVC
VisualStoryTelling | 16,752 | 33,794 | VST
Ego4D | xxx | xxx | E4D
CocoSpotTheDifference| coming soon | coming soon | CSD

## Download Links

| Dataset | Download Link |
| :--- | :--- |
| coreset | |

## Syphus Overview

<p align="center" width="100%">
<img src="https://i.postimg.cc/YSSmVCc9/syphus.png"  width="80%" height="80%">
</p>

Syphus, an automated pipeline for generating high-quality instruction-response pairs in multiple languages. Building upon the framework proposed by LLaVA, we utilize ChatGPT to generate instruction-response pairs based on visual content. To ensure the quality of the generated instruction-response pairs, our pipeline incorporates system messages, visual annotations, and in-context examples as prompts for ChatGPT. System messages define the desired tone and style of the generated instruction-response pairs, while visual annotations provide essential image information such as bounding boxes and image descriptions. In-context examples assist ChatGPT in learning within the context.  During cold-start stage, in-context examples are collected by prompting ChatGPT solely through system messages and visual annotations, employing a heuristic approach. This stage concludes only when a satisfactory in-context examples are identified. In step 4, once the instruction-response pairs are obtained, the pipeline expands them into Chinese (zh), Japanese (ja), Spanish (es), German (de), French (fr), Korean (ko), and Arabic (ar).

## Syphus on your own dataset

We provide source code of the framework in [syphus](Syphus) folder. You can use it to generate instruction-response pairs on your own dataset following the steps below.

1. Configure openai key. Create the following environment variables in your system.

``` bash
export OPENAI_API_TYPE="xxx"
export OPENAI_API_KEY="xxxxxxxxxxxxxxxxxxx"
export OPENAI_API_BASE="https://xxxx"
export OPENAI_API_VERSION="2023-03-15-xx"
export OPENAI_API_ENGINE="chatgpt0301"
```

1. Create a new file in `datasets` folder and name it as `your_dataset.py`.
2. Create a class named `YourDataset` in `your_dataset.py` and inherit from `abstract_dataset.AbstractDataset` class.
3. Implement `_load_query_inputs` methods in `YourDataset` class. This method should return a list of dict, each dict contains `id` and `sentences` keys.
   1. `id` is the unique id of each query
   2.  `sentences` is a string for each query input.
4.  Define `system_message`, `in-context example` in `prompts` folder.
5.  Define `_load_prefix` methods in `YourDataset` class. 
    1.  This method should return a dictionary. The keys of the dictionary are `system_message`, `in_context`.
    2.  The value of `system_message` is a string.
    3.  The value of `in_context` is a list of `{"role": role, "content": content_string}`.
6. You are done! Run the following command to generate instruction-response pairs on your own dataset.

``` bash
python  main.py --name YourDataset.your_dataset --num_threads 4
```

## 📝 Citation