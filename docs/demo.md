## 🪩 Serving Demo

### Launch a controller

```Shell
python -m pipeline.serve.controller --host 0.0.0.0 --port 10000
```

### Launch a model worker

```Shell
# Init our model on GPU
CUDA_VISIBLE_DEVICES=0,1 python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model_name otter --checkpoint_path checkpoint/multi_instruct_chunyuan-core_otter9B_lr1e-5_6epochs_hf --num_gpus 2

# Init original open flamingo model on GPU
CUDA_VISIBLE_DEVICES=2,3 python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo_original --checkpoint_path luodian/openflamingo-9b-hf --num_gpus 2

# Init original open flamingo model on CPU
python -m pipeline.serve.model_worker --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model_name open_flamingo_original --checkpoint_path luodian/openflamingo-9b-hf --num_gpus 0
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...".

### Launch a gradio web server

```Shell
python -m pipeline.serve.gradio_web_server --controller http://localhost:10000
```

#### You can open your browser and chat with a model now.