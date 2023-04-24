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

``` bash
python -m torch.distributed.run --nproc_per_node=1 open_flamingo/train/instruction_following.py \
--run_name=flamingo3B \
--lm_path=facebook/opt-1.3b \
--tokenizer_path=facebook/opt-1.3b \
--dataset_resampled \
--multi_instruct_path=./example_multi_instruct_data/vision_language_examples.tsv \
--batch_size=8 --num_epochs=30 \
--report_to_wandb --wandb_project=flamingo3B \
--wandb_entity=drluodian \
--delete_previous_checkpoint \
--run_name=dev/multi_instruct_caption_flamingo3B 
```

### Hyperparameters

## Experiments Results

We report accuracy on following datasets after instruction following (IF) tuning. 

1. COCO Caption
2. VQAv2
3. OKVQA
4. ImageNet
5. Flickr30

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

### Serving Demo
#### Launch a controller
```Shell
python -m collie_core.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a model worker
```Shell
export AZURE_DIR="/media/ntu/volume1/home/s121md302_06/data/data/azure"
# Init our model on GPU
python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-name open_flamingo --checkpoint-path models/collie_llama9b_multi_instruct_apr23/final_weights.pt --num-gpus 1 
# Init our model on CPU
python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-name open_flamingo --checkpoint-path ${AZURE_DIR}/models/collie_llama9b_multi_instruct_apr23/final_weights.pt --num-gpus 0

# Init original model on CPU
python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model-name open_flamingo_original --num-gpus 0
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".

#### Send a test message
```Shell
python -m collie_core.serve.test_message --model-name LLaVA-13B-v0 --controller http://localhost:10000
```

#### Launch a gradio web server.
```Shell
python -m collie_core.serve.gradio_web_server --controller http://localhost:10000
```
#### You can open your browser and chat with a model now.

### Authors

Equal contribution, alphabetical order
