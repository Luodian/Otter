# Demo Hosting

We will show you how to host a demo on your own computer using gradio.

## Preparation

### Download the checkpoints

We assume that you have downloaded the 🦦 Otter checkpoint and the 🦩 Open Flamingo checkpoint.

### Install additional packages

``` bash
pip install fastapi
```

## Launch a controller

``` bash
python -m collie_core.serve.controller --host 0.0.0.0 --port 10000
```

## Launch a model worker

Initialize our 🦦 Otter model on GPU

``` bash

CUDA_VISIBLE_DEVICES=0,1 python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model_name collie --checkpoint_path path/to/the/otter/checkpoint --num_gpus 2

```

Initialize the 🦩 Open Flamingo model on GPU

``` bash
CUDA_VISIBLE_DEVICES=2,3 python -m collie_core.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo_original --checkpoint_path path/to/the/flamingo/checkpoint --num_gpus 2
```

Wait until the process finishes loading the model then you will see "Uvicorn running on ...".

## Send a test message

```Shell
python -m collie_core.serve.test_message --model_name LLaVA-13B-v0 --controller http://localhost:10000
```

## Launch a gradio web server

```Shell
python -m collie_core.serve.gradio_web_server --controller http://localhost:10000
```

## Now, you can open your browser and chat with the model!
