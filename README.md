<p align="center" width="100%"><img src="assets/logo.png" alt="ntu-otter" style="width: 100%; min-width: 300px; display: block; margin: auto;"></a>
</p>

#
## Talk to 🦦 Otter, Give it instructions, and Explore its in-context learning ability!

[Otter Demo](https://otter.cliangyu.com/)

## Overview

Recent research has highlighted the significance of instruction tuning in enabling Large Language Models (LLMs) to effectively carry out real-world tasks and adhere to natural language instructions. For example, GPT-3 can be improved to become Chat-GPT through instruction tuning. The Flamingo project is considered to be a milestone for GPT-3 in the multimodal domain.

In our project, we introduce 🦦 Otter, an in-context instruction-tuning model that builds on Flamingo. To enhance its capabilities, we leverage a multimodal instruction tuning dataset. Each data sample comprises an image-specific instruction as well as multiple multimodal instructions, also known as multimodal in-context learning examples.

## Environment

You may install via `pip install -r requirements.txt`  or manually install the following packages.
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
conda install -c conda-forge tensorboard=2.12.0 -y
conda install -c conda-forge more-itertools=9.1.0 -y

# install standford-corenlp-full
cd LAVIS/coco-caption;
sh get_stanford_models.sh
```

</details>

## Dataset

### Details

For details of our training data,  check our [dataset card](/docs/dataset_card.md).

### Preparation

Download a subset of the pretraining `multi_instruct` dataset

```bash
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/datasets/pretrain_data/pretrain_data_examples.zip;
unzip pretrain_data_examples.zip ./example_multi_instruct_data
```

## Training

Train on `multi_instruct` example datasets, use following commands:

## Model

### Model Details

For details of 🦦 Otter, you may check our [model card](/docs/model_card.md).

The trained checkpoints will come soon.

### 🤗 Flamingo Hugging Face Model

For future research, we rewrite the 🦩 Flamingo model to hugging face format. Right now, you can use the 🦩 Flamingo model as a hugging face model with only two lines!

``` python
from flamingo_hf import FlamingoModel
model = FlamingoModel.from_pretrained("luodian/openflamingo-9b-hf")
```

## Serving Demo

For hosting the 🦦 Otter on your own computer, follow the [demo instructions](/docs/dataset_card.md).

## Authors

Equal contribution, alphabetical order.

[Liangyu Chen](https://cliangyu.com/)

[Bo Li](https://brianboli.com/)

[Jinghao Wang](https://king159.github.io/)

[Yuanhan Zhang](https://zhangyuanhan-ai.github.io/)

## Acknowledgements

We would like to extend our gratitude to Jingkang Yang and Ziwei Liu for their invaluable assistance and unwavering support. Additionally, we would like to express our appreciation to the Open Flamingo team for their exceptional contributions to the open source community.
