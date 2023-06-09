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

[Project Page](https://otter-ntu.github.io/) | [Youtube Video](https://www.youtube.com/watch?v=K8o_LKGQJhs) | [Bilibili Video](https://www.bilibili.com/video/BV1Bo4y1T7SN/?share_source=copy_web&vd_source=477facaaaa60694f67a784f5eaa905ad) | 📝[Paper]()

## 🌳 MIMIC-IT Overview

MIMIC-IT covers a vast array of real-life scenarios that empower Vision-Language Models (VLMs) to not only comprehend general scenes, but also to reason about context and astutely differentiate between observations. MIMIC-IT also enables the application of egocentric visual assistant model that can serve that can answer your questions like **Hey, Do you think I left my keys on the table?**. In addition to English, MIMIC-IT is also multilingual, supporting Chinese, Korean, Japanese, German, French, Spanish, and Arabic, thereby allowing a larger global audience to altogether enjoy from the convenience brought about by advancements in artificial intelligence.

High-quality instructions are essential for the zero-shot performance of large language models on interactive natural language tasks. For interactive vision-language tasks involving intricate visual scenes, a large quantity of diverse and creative instructions should be imperative to tune vision-language models (VLMs). Nevertheless, the current availability of vision-language instructions in terms of quantity, diversity, and creativity remains limited, posing challenges to the generalization of interactive VLMs. Here we present **MIMIC-IT**, a dataset comprising 2.8M multi-modal instructions-response pairs based on images and videos. Each instruction-response pair is accompanied by multi-modal in-context information, forming conversational contexts aimed at empowering VLMs in perception, reasoning, and planning. The instruction-response collection process, dubbed as $Syphus$, is scaled using an automatic annotation pipeline that combines human expertise with GPT's capabilities.

<p align="center" width="100%">
<img src="https://i.postimg.cc/k406BN26/mimic-it.png"  width="80%" height="80%">
</p>

## Dataset Statistics

| **Visual Sources (Scenes)** | **In-context** | **#Clips/Images** | **#Uni. Inst.** | **#Instances** |  
|---------------------------|----------------|------------------|-----------------|----------------|  
| COCO (General) | lang./vis. | - / 81K | 261K | 345K |  
| SD~(Surveillance) | lang./vis. | - / 9K | 10K | 15K  |  
| SN~(Indoor Ego.) | lang./vis. | - / 0.5K | 4.8K | 6K  |  
| DC~(General) | lang./vis. | 16K / 1M | 40K | 62K  |  
| VIST~(Story) | lang./vis. | - / 16K | 32K | 33K  |  
| TVC~(TV) | lang./vis. | 86K / 577K | 86K | 92K  |  
| E4D~(General Ego.) | lang./vis. | 400K / 6.4M | 1.8M | 2.4M  |  
| Total | lang./vis. | 502K / 8.1M | 2.2M | 2.8M |  

## Download Links

We are organizing the conversion of public dataset images (as well as extracting specific frames from the corresponding videos) to the MIMIC-IT input format. Below are the initial released LA and DC, instruction-response pairs for the MIMIC-IT dataset. Instruction pairs on other datasets are larger and contain more information that may need further examination. We will release this data as soon as possible.

| Scenes | Images/Videos | Annotations |
| :--- | :--- | :--- |
| **LA In-context** | Processing | |
| **Dense Caption** | Processing | |

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