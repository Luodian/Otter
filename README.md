<p align="center" width="100%">
<img src="https://i.postimg.cc/MKmyP9wH/new-banner.png"  width="80%" height="80%">
</p>

<div>
<div align="center">
    <a href='https://brianboli.com/' target='_blank'>Bo Li<sup>*,♠,1</sup></a>&emsp;
    <a href='https://zhangyuanhan-ai.github.io/' target='_blank'>Yuanhan Zhang<sup>*,♠,1</sup></a>&emsp;
    <a href='https://cliangyu.com/' target='_blank'>Liangyu Chen<sup>*,1</sup></a>&emsp;
    <a href='https://king159.github.io/' target='_blank'>Jinghao Wang<sup>*,1</sup></a>&emsp;
    <a href='https://pufanyi.github.io/' target='_blank'>Fanyi Pu<sup>*,1</sup></a>&emsp;
    </br>
    <a href='https://jingkang50.github.io/' target='_blank'>Jingkang Yang<sup>1</sup></a>&emsp;
    <a href='https://chunyuan.li/' target='_blank'>Chunyuan Li<sup>2</sup></a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>&#x2709,1</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp;
    <sup>2</sup>Microsoft Research, Redmond
    </br>
    <sup>♠</sup> Co-Project Lead&emsp;
    <sup>*</sup> Equal Contribution&emsp;
    <sup>&#x2709</sup> Corresponding Author
    
</div>
 
 -----------------

![](https://img.shields.io/badge/otter-v0.2-darkcyan)
![](https://img.shields.io/github/stars/luodian/otter?style=social)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLuodian%2Fotter&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
![](https://img.shields.io/badge/code%20style-black-000000.svg)

[Project Page](https://otter-ntu.github.io/) | [Otter Paper](https://arxiv.org/abs/2305.03726) | [MIMIC-IT Paper](https://arxiv.org/abs/2306.05425) | [MIMIC-IT Dataset](mimic-it/README.md)

**Video Demo:** [Otter's Conceptual Demo Video](https://www.youtube.com/watch?v=K8o_LKGQJhs) | [Bilibili 哔哩哔哩](https://www.bilibili.com/video/BV1Bo4y1T7SN/?share_source=copy_web&vd_source=477facaaaa60694f67a784f5eaa905ad)

**Interactive Demo:**  [Otter-Image](https://otter.cliangyu.com/) | [Otter-Video](https://ottervideo.cliangyu.com/)
> Our models would be sometimes offline due to GPU limitation (if we need to train new models lol). You can refer to 🏎️ [Run Otter Locally](./pipeline/demo) to try Otter-Image and Otter-Video more smoothly on your local machine, with at least 16G GPU mem (BF16/FP16 Mode) to help your tasks like image/video tagging, captioning or identifying harmful content.

**Corresponding Checkpoints:**  [luodian/OTTER-Image-MPT7B](https://huggingface.co/luodian/OTTER-Image-MPT7B) | [luodian/OTTER-Video-LLaMA7B-DenseCaption](https://huggingface.co/luodian/OTTER-Video-LLaMA7B-DenseCaption)
> **Otter-Image** supports multiple images input as in-context examples, which is **the first multi-modal instruction tuned model** that supports to organize inputs this way.

> **Otter-Video** supports videos inputs (frames are arranged as original Flamingo's implementation) and multiple images inputs (they serve as in-context examples for each other).

**Eval Results:** [Multi-Modal Arena](http://vlarena.opengvlab.com/) | [MLLM Evaluation Benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | [OpenCompass-MMBench](https://opencompass.org.cn/leaderboard-multimodal)

## 🦾 Update

**Contact: Leave issue or `drluodian@gmail.com/YUANHAN002@e.ntu.edu.sg`. We are on call to respond.**

**[2023-07]**
1. 🧨 Feature Updates:
    - DeepSpeed ZeRo2 Integration + DDP Training
    - Support Flamingo pretraining on Laion400M/CC3M.
    - Add LoRA support for tuning LLM decoder.
    - Integration of multiple LLMs (Vicuna, MPT, LLama2, Falcon)
2. 🤗 Checkout [MIMIC-IT](https://huggingface.co/datasets/pufanyi/MIMICIT) on Huggingface datasets.
1. 🦦 Checkout our [Otter-MPT7B Image Demo](https://otter.cliangyu.com/). We update the model by incoporating OpenFlamingv2 and specifically tune it to enable generation abilities for both long and short answers.
2. 🥚 Update [Eggs](./mimic-it/README.md/#eggs) section for downloading MIMIC-IT dataset.
3. 🥃 Contact us **if you wish to develop Otter for your scenarios** (for satellite images or funny videos?). We aim to support and assist with Otter's diverse use cases. OpenFlamingo and Otter are strong models with the [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)'s excellently designed architecture that accepts multiple images/videos or other modality inputs. Let's build more interesting models together. 

**[2023-06]**
1. 🧨 [Download MIMIC-IT Dataset](https://entuedu-my.sharepoint.com/:f:/g/personal/libo0013_e_ntu_edu_sg/Eo9bgNV5cjtEswfA-HfjNNABiKsjDzSWAl5QYAlRZPiuZA?e=M9isDT). For more details on navigating the dataset, please refer to [MIMIC-IT Dataset README](mimic-it/README.md).
2. 🏎️ [Run Otter Locally](./pipeline/demo). You can run our model locally with at least 16G GPU mem for tasks like image/video tagging and captioning and identifying harmful content. We fix a bug related to video inference where `frame tensors` were mistakenly unsqueezed to a wrong `vision_x`.
    > Make sure to adjust the `sys.path.append("../..")` correctly to access `otter.modeling_otter` in order to launch the model.
3. 🏇 We welcome third-party evaluation on Otter and we are willing to see different VLMs chasing with each other on different arenas and benchmarks. But make sure contact us to confirm the model version and prompt strategy before publishing results. We are on call to respond.
4. 🤗 Introducing Project Otter's brand new homepage: https://otter-ntu.github.io/. Check it out now!
5. 🤗 Check our [paper](https://arxiv.org/abs/2306.05425) introducing MIMIC-IT in details. Meet MIMIC-IT, the first multimodal in-context instruction tuning dataset with 2.8M instructions! From general scene understanding to spotting subtle differences and enhancing egocentric view comprehension for AR headsets, our MIMIC-IT dataset has it all.
<!-- 6. 🤗 Stay tuned for our upcoming Otter Model v0.2, trained on the MIMIC-IT dataset! With the ability to understand daily scenes, reason in context, spot differences in observations, and act as an egocentric assistant. Checkout conceptual demo video at [Youtube](https://www.youtube.com/watch?v=K8o_LKGQJhs) or [Bilibili](https://www.bilibili.com/video/BV1Bo4y1T7SN/?share_source=copy_web&vd_source=477facaaaa60694f67a784f5eaa905ad)! -->

<!-- **[2023-05-14]**
1. Otter battles with Owl? the Pokémon Arena is here! Our model is selected into [Multi-Modal Arena](http://vlarena.opengvlab.com/). This is an interesting Multi-Modal Foundation Models competition arena that let you see different models reaction to the same question.

**[2023-05-08]**
1. Check our Arxiv release paper at [Otter: A Multi-Modal Model with In-Context Instruction Tuning](https://arxiv.org/abs/2305.03726) !
2. We support `xformers` for memory efficient attention. -->

<div style="text-align:center">
<img src="https://i.postimg.cc/Tw1Z0BCW/otterv0-2-demo.png"  width="100%" height="100%">
</div>

## 🦦 Why In-Context Instruction Tuning?

Large Language Models (LLMs) have demonstrated exceptional universal aptitude as few/zero-shot learners for numerous tasks, owing to their pre-training on extensive text data. Among these LLMs, GPT-3 stands out as a prominent model with significant capabilities. Additionally, variants of GPT-3, namely InstructGPT and ChatGPT, have proven effective in interpreting natural language instructions to perform complex real-world tasks, thanks to instruction tuning. 

Motivated by the upstream interleaved format pretraining of the Flamingo model, we present 🦦 Otter, a multi-modal model based on OpenFlamingo (the open-sourced version of DeepMind's Flamingo). We train our Otter in an in-context instruction tuning way on our proposed **MI**-**M**odal **I**n-**C**ontext **I**nstruction **T**uning (**MIMIC-IT**) dataset. Otter showcases improved instruction-following and in-context learning ability in both images and videos.

## 🗄 MIMIC-IT Dataset Details

<p align="center" width="100%">
<img src="https://i.postimg.cc/yYMm1G5X/mimicit-logo.png"  width="80%" height="80%">
</p>

MIMIC-IT enables the application of egocentric visual assistant model that can serve that can answer your questions like **Hey, Do you think I left my keys on the table?**. Harness the power of MIMIC-IT to unlock the full potential of your AI-driven visual assistant and elevate your interactive vision-language tasks to new heights.

<p align="center" width="100%">
<img src="https://i.postimg.cc/RCGp0vQ1/syphus.png"  width="80%" height="80%">
</p>

We also introduce **Syphus**, an automated pipeline for generating high-quality instruction-response pairs in multiple languages. Building upon the framework proposed by LLaVA, we utilize ChatGPT to generate instruction-response pairs based on visual content. To ensure the quality of the generated instruction-response pairs, our pipeline incorporates system messages, visual annotations, and in-context examples as prompts for ChatGPT. 

For more details, please check the [MIMIC-IT dataset](mimic-it/README.md).


## 🤖 Otter Model Details

<div style="text-align:center">
<img src="https://i.postimg.cc/CKgQ2PP7/otter-teaser.png"  width="100%" height="100%">
</div>

Otter is designed to support multi-modal in-context instruction tuning based on the OpenFlamingo model, which involves conditioning the language model on the corresponding media, such as an image that corresponds to a caption or an instruction-response pair.

We train Otter on MIMIC-IT dataset with approximately 2.8 million in-context instruction-response pairs, which are structured into a cohesive template to facilitate various tasks. Otter supports videos inputs (frames are arranged as original Flamingo's implementation) and multiple images inputs as in-context examples, which is **the first multi-modal instruction tuned model**. 

The following template encompasses images, user instructions, and model-generated responses, utilizing the `User` and `GPT` role labels to enable seamless user-assistant interactions.

```python
prompt = f"<image>User: {instruction} GPT:<answer> {response}<endofchunk>"
```

Training the Otter model on the MIMIC-IT dataset allows it to acquire different capacities, as demonstrated by the LA and SD tasks. Trained on the LA task, the model exhibits exceptional scene comprehension, reasoning abilities, and multi-round conversation capabilities. 

```python
# multi-round of conversation
prompt = f"<image>User: {first_instruction} GPT:<answer> {first_response}<endofchunk>User: {second_instruction} GPT:<answer>"
```

Regarding the concept of organizing visual-language in-context examples, we demonstrate here the acquired ability of the Otter model to follow inter-contextual instructions after training on the LA-T2T task. The organized input data format is as follows:

```python
# Multiple in-context example with similar instructions
prompt = f"<image>User:{ict_first_instruction} GPT: <answer>{ict_first_response}<|endofchunk|><image>User:{ict_second_instruction} GPT: <answer>{ict_second_response}<|endofchunk|><image>User:{query_instruction} GPT: <answer>"
```

For more details, please refer to our [paper](https://arxiv.org/abs/2306.05425)'s appendix for other tasks.
## 🗂️ Environments

1. Compare cuda version returned by nvidia-smi and nvcc --version. They need to match. Or at least, the version get by nvcc --version should be <= the version get by nvidia-smi.
2. Install the pytorch that matches your cuda version. (e.g. cuda 11.7 torch 2.0.0). We have successfully run this code on cuda 11.1 torch 1.10.1 and cuda 11.7 torch 2.0.0. You can refer to PyTorch's documentation, [Latest](https://pytorch.org/) or [Previous](https://pytorch.org/get-started/previous-versions/).
3. You may install via `conda env create -f environment.yml`. Especially to make sure the `transformers>=4.28.0`, `accelerate>=0.18.0`.

After configuring environment, you can use the 🦩 Flamingo model / 🦦 Otter model as a 🤗 Hugging Face model with only a few lines! One-click and then model configs/weights are downloaded automatically. Please refer to [Huggingface Otter/Flamingo](./docs/huggingface_compatible.md) for details.

## ☄️ Training

Otter is trained based on OpenFlamingo. You may need to use converted weights at [luodian/OTTER-9B-INIT](https://huggingface.co/luodian/OTTER-9B-INIT) or [luodian/OTTER-MPT7B-Init](https://huggingface.co/luodian/OTTER-MPT7B-Init). They are respectively converted from [OpenFlamingo-LLaMA7B-v1](https://huggingface.co/openflamingo/OpenFlamingo-9B) and [OpenFlamingo-MPT7B-v2](https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b), we added a `<answer>` token for Otter's downstream instruction tuning. 

You may also use any trained Otter weights to start with your training on top of ours, see them at [Otter Weights](https://huggingface.co/luodian). You can refer to [MIMIC-IT](https://github.com/Luodian/Otter/tree/main/mimic-it) for preparing image/instruction/train json files.

```bash
export PYTHONPATH=.

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_fsdp.yaml \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path=luodian/OTTER-LLaMA7B-INIT  \ # or --pretrained_model_name_or_path=luodian/OTTER-MPT7B-Init
--mimicit_path="path/to/DC_instruction.json" \
--images_path="path/to/DC.json" \
--train_config_path="path/to/DC_train.json" \
--batch_size=4 \
--num_epochs=9 \
--report_to_wandb \
--wandb_entity=ntu-slab \
--run_name=OTTER-LLaMA7B-densecaption \
--wandb_project=OTTER-LLaMA7B \
--workers=1 \
--lr_scheduler=cosine \
--learning_rate=1e-5 \
--warmup_steps_ratio=0.01 \
```

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
    title={MIMIC-IT: Multi-Modal In-Context Instruction Tuning},
    author={Bo Li and Yuanhan Zhang and Liangyu Chen and Jinghao Wang and Fanyi Pu and Jingkang Yang and Chunyuan Li and Ziwei Liu},
    year={2023},
    eprint={2306.05425},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

### 👨‍🏫 Acknowledgements

We thank [Jack Hessel](https://jmhessel.com/) for the advise and support, as well as the [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) team for their great contribution to the open source community.

Huge accolades to [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) and [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) team for the work on this great architecture.

### 📝 Related Projects

- [LLaVA: Visual Instruction Tuning](https://github.com/haotian-liu/LLaVA)
- [Instruction Tuning with GPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
