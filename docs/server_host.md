## 🪩 Serving Demo

We will show you how to host a demo on your own computer using gradio.

## Preparation

### Download the checkpoints

The 🦦 Otter checkpoint and the 🦩 Open Flamingo checkpoint can be auto-downloaded with the code below.

## Start Demo 

### Launch a controller

```Shell
python -m pipeline.serve.controller --host 0.0.0.0 --port 10000
```

### Launch a model worker

```Shell
# Init our 🦦 Otter model on GPU
CUDA_VISIBLE_DEVICES=0,1 python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model_name otter --checkpoint_path luodian/otter-9b-hf --num_gpus 2 --limit_model_concurrency 200
# Init our 🦦 Otter video model on CPU
CUDA_VISIBLE_DEVICES=0,1 python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40002 --worker http://localhost:40002 --model_name otter_video --checkpoint_path checkpoint/otter9B_DC_fullset_16frames/ --num_gpus 2 --limit_model_concurrency 200 --load_bit 16
# Init original open flamingo model on GPU
CUDA_VISIBLE_DEVICES=2,3 python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo --checkpoint_path luodian/openflamingo-9b-hf --num_gpus 2 --limit_model_concurrency 200

# Init original open flamingo model on CPU
python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo_original --checkpoint_path luodian/openflamingo-9b-hf --num_gpus 0
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...".

### Launch a gradio web server

```Shell
# Image demo
python -m pipeline.serve.gradio_web_server --controller http://localhost:10000 --port 7861
# Video demo
python -m pipeline.serve.gradio_web_server_video --controller http://localhost:10000 --port 7862
```

Now, you can open your browser and chat with the model!

### Examples
If you encounter error stating `FileNotFoundError: [Errno 2] No such file or directory: '/home/luodian/projects/Otter/pipeline/serve/examples/Apple Vision Pro - Reveal Trailer.mp4'`

That's because we didnt upload the video examples on Github. You could visit the following [folder](https://entuedu-my.sharepoint.com/:f:/g/personal/libo0013_e_ntu_edu_sg/EjjDhJm4G35EgVHo0Pxi7dEBM7rqdN3e0ZcBCskWuIubUQ?e=C58jI3) to download our used examples and put them to the right place.
