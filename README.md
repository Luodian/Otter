<header><img src="./assets/collie_icon.png" alt="Collie Icon"><h1>Collie: A Visual Language Model with Efficient Instruction Tuning</h1></header>

Collie interprets and deciphers complex visual information, enabling seamless integration of images and text. Collie is built on OpenFlamingo.

## Original LAVIS

if something goes wrong, please checkout to `original_lavis` branch to conduct test. Some errors actually exist in original lavis repo.

## Overview

PET-VLM project aims to finetune a Large Visual Language Model (VLM) on downstream tasks. We use the OpenFlamingo-9B using a CLIP ViT-Large vision encoder and a LLaMA-7B language model.

## Fine-tuning

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
pip install open_clip_torch==2.16.0
pip install einops-exts==0.0.4
conda install -c conda-forge tensorboard=2.12.0 -y
conda install -c conda-forge more-itertools=9.1.0 -y
conda install -c conda-forge black=23.3.0 -y
pip install gpustat

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

### Authors

Equal contribution, alphabetical order
