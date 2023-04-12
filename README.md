# PET-VLM

## Original LAVIS

if something goes wrong, please checkout to `original_lavis` branch to conduct test. Some errors actually exist in original lavis repo.

## Overview

PET-VLM project aims to finetune a Large Visual Language Model (VLM) on downstream tasks. We use the OpenFlamingo-9B using a CLIP ViT-Large vision encoder and a LLaMA-7B language model. We finetune the VLM on VQA and COCO dataset.

## Fine-tuning

### Environment

You may install via `conda create -f environment.yaml` or manually install the following packages.
<details>
<summary>Manually Install</summary>

``` bash
conda install pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c conda-forge datasets=2.11.0 -y
conda install -c conda-forge wandb=0.14.0 -y
conda install -c conda-forge braceexpand=0.1.5 -y
conda install -c conda-forge webdataset=0.2.48 -y
conda install -c conda-forge scipy=1.10.1 -y
conda install -c conda-forge sentencepiece=0.1.97 -y
conda install -c conda-forge einops=0.6.0 -y
pip install bitsandbytes==0.37.2
pip install open_clip_torch==2.16.0
pip install einops-exts==0.0.4
conda install -c conda-forge tensorboard=2.12.0 -y
conda install -c conda-forge more-itertools=9.1.0 -y
pip install git+https://github.com/huggingface/transformers
conda install -c conda-forge black=23.3.0 -y
pip install gpustat

# install standford-corenlp-full
cd LAVIS/coco-caption;
sh get_stanford_models.sh
```

</details>

### Training

Our model is trained on 4 V100 GPUs. You may use the following command to train the model.

``` bash
# demo run on finetuning coco_caption
torchrun --nproc_per_node=4 LAVIS/train.py --cfg-path=LAVIS/lavis/projets/peft_flamingo/caption_coco_ft.yaml
```

### Hyperparameters

## Experiments Results

### VQAv2 (VQA accuracy)

|            | 0-shot | 4-shot | 8-shot | 16-shot | 32-shot |
|------------|--------|--------|--------|---------|---------|
| OpenFlamingo-9B (ViT-G + ) | 43.5   | 44.0   | 47.5   | 48.9    | 50.3    |
| DeepMind Flamingo-9B | 51.8   | 56.3   | 58.0   | 59.4    | 60.41   |

### COCO Caption (CIDEr)

|            | 0-shot | 4-shot | 8-shot | 16-shot | 32-shot |
|------------|--------|--------|--------|---------|---------|
| OpenFlamingo-9B | 65.5   | 74.3   | 79.3   | 81.8    | 84.5    |
| DeepMind Flamingo-9B | 79.4   | 93.1   | 99.0   | 102.2   | 106.3   |

### Authors

Equal contribution, alphabetical order

[Li Bo](https://brianboli.com/)

[Wang Jinghao](https://king159.github.io/)
