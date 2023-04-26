<header><img src="./assets/collie_icon.png" alt="Collie Icon"><h1>🐾 Collie: A Visual Language Model with Efficient Instruction Tuning</h1></header>

Collie interprets and deciphers complex visual information, enabling seamless integration of images and text. Collie is built on OpenFlamingo.

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


### Serving Demo

#### Convert model into HF
```Shell
export AZURE_DIR="/media/ntu/volume1/home/s121md302_06/data/data/azure"
export PYTHONPATH=.
# Convert our model into HF, add <answer> token
python flamingo_hf/converting_flamingo_to_pytorch.py --old ${AZURE_DIR}/models/collie_llama9b_multi_instruct_apr23/final_weights.pt --new checkpoint/collie_llama9b_multi_instruct_apr23/final_weights_hf.pt
python flamingo_hf/save_hf.py --checkpoint checkpoint/collie_llama9b_multi_instruct_apr23/final_weights_hf.pt --save-dir checkpoint/collie_llama9b_multi_instruct_apr23_hf --add-answer-token
# Convert official flamingo model into HF, don't add <answer> token
python flamingo_hf/converting_flamingo_to_pytorch.py --old /media/ntu/volume1/home/s121md302_06/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-9B/snapshots/b5cd34cb6c90775b262837b6a80a6a47123b4571/checkpoint.pt --new checkpoint/models--openflamingo--OpenFlamingo-9B/checkpoint.pt
python flamingo_hf/save_hf.py --checkpoint checkpoint/models--openflamingo--OpenFlamingo-9B/checkpoint.pt --save-dir checkpoint/open_flamingo_9B_hf
```

#### Launch a controller
```Shell
python -m collie_core.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a model worker
```Shell
export AZURE_DIR="/media/ntu/volume1/home/s121md302_06/data/data/azure"
# Init our model on GPU
CUDA_VISIBLE_DEVICES=0,1 python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model_name collie_llama9b_multi_instruct_apr23 --checkpoint_path checkpoint/collie_llama9b_multi_instruct_apr23_hf --num_gpus 2

# Init original model on GPU
CUDA_VISIBLE_DEVICES=2,3 python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo_original --checkpoint_path checkpoint/open_flamingo_9B_hf --num_gpus 2
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".

#### Send a test message
```Shell
python -m collie_core.serve.test_message --model_name LLaVA-13B-v0 --controller http://localhost:10000
```

#### Launch a gradio web server.
```Shell
python -m collie_core.serve.gradio_web_server --controller http://localhost:10000
```
#### You can open your browser and chat with a model now.

### Authors

Equal contribution
