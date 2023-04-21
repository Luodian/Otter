"""
A model worker executes the model.
"""
import argparse
import asyncio
import dataclasses
import logging
import json
import time
from typing import List, Union
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from functools import partial

from collie_core.constants import WORKER_HEART_BEAT_INTERVAL
from collie_core.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from transformers import LlamaForCausalLM, AutoModelForCausalLM

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_DEMO_END_TOKEN = "<|endofchunk|>"


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_name,
                 is_multi_modal, keep_aspect_ratio,
                 num_gpus):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.is_multi_modal = is_multi_modal
        self.keep_aspect_ratio = keep_aspect_ratio
        self.tokenizer, self.model, self.image_processor, self.context_len = self.load_model(
            model_path, num_gpus, is_multi_modal)

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()
    
    def load_model(self, model_path, num_gpus, is_multi_modal):
        # hard code model initialization
        flamingo_lm_path = "decapoda-research/llama-7b-hf"
        
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=flamingo_lm_path,
            tokenizer_path=flamingo_lm_path,
            cross_attn_every_n_layers=4
        )
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        if num_gpus == 1:
            self.device = 'cuda'
            model.cuda()
        elif num_gpus > 1:
            self.device = 'cuda'
            raise NotImplementedError("Multi-GPU is not supported yet.")
        else:
            self.device = 'cpu'
        
        # if hasattr(model.config, "max_sequence_length"):
        #     context_len = model.config.max_sequence_length
        # else:
        #     context_len = 2048
        context_len = 2048
        
        return tokenizer, model, image_processor, context_len

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=25)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        if images is not None:
            from PIL import Image
            from io import BytesIO
            import base64
            assert type(images) is list
            if len(images) > 0:
                images = [Image.open(BytesIO(base64.b64decode(image))) for image in images]
                assert len(images) == prompt.count(DEFAULT_IMAGE_TOKEN), "Number of images does not match number of <image> tokens in prompt"
                    
                images = [image_processor(image).unsqueeze(0) for image in images]
                vision_x = torch.cat(images, dim=0).unsqueeze(1).unsqueeze(0).to(self.device)
            else:
                images = None

        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)

        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)
        pred_ids = []

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        past_key_values = None
        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    lang_x=torch.as_tensor([input_ids]).to(self.device), 
                    vision_x=vision_x,
                    clear_conditioned_layers=False,
                    use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device=self.device)
                out = model(vision_x=None,
                            lang_x=torch.as_tensor([[token]], device=self.device),
                            use_cached_vision_x=True,
                            use_cache=True,
                            attention_mask=attention_mask,
                            clear_conditioned_layers=False,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            pred_ids.append(token)

            if token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % args.stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                output = tokenizer.decode(output_ids, skip_special_tokens=True)

                if images is not None:
                    # HACK: deal with images
                    cur_out = tokenizer.decode(pred_ids, skip_special_tokens=True)
                    pos = cur_out.rfind(stop_str)
                    if pos != -1:
                        cur_out = cur_out[:pos]
                        stopped = True
                    output = ori_prompt + cur_out
                else:
                    pos = output.rfind(stop_str, l_prompt)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True

                ret = {
                    "text": output,
                    "error_code": 0,
                }
                yield json.dumps(ret).encode() + b"\0"

            if stopped:
                break

        if past_key_values is not None:
            del past_key_values

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--multi-modal", action="store_true")
    parser.add_argument("--keep-aspect-ratio", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_name,
                         args.multi_modal,
                         args.keep_aspect_ratio,
                         args.num_gpus)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
