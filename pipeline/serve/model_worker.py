"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
from PIL import Image
from io import BytesIO
import base64


from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
from transformers import TextIteratorStreamer
import torch
import uvicorn
from functools import partial

from pipeline.constants import WORKER_HEART_BEAT_INTERVAL
from pipeline.serve.serving_utils import (
    build_logger,
    server_error_msg,
    pretty_print_semaphore,
)
from huggingface_hub import hf_hub_download
import transformers
from otter import OtterForConditionalGeneration
from flamingo import FlamingoForConditionalGeneration

GB = 1 << 30

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
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        no_register,
        lm_path,
        model_name,
        checkpoint_path,
        keep_aspect_ratio,
        num_gpus,
        load_bit,
        load_pt,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_name = model_name
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.keep_aspect_ratio = keep_aspect_ratio
        self.load_bit = load_bit
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = self.load_model(lm_path, checkpoint_path, num_gpus, load_pt)

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def load_model(self, lm_path, checkpoint_path, num_gpus, load_pt=None):
        # if not load_pt:
        device_map = "auto" if num_gpus > 0 else None
        if self.load_bit == "int8":
            precision = {"load_in_8bit": True}
        elif self.load_bit == "int4":
            precision = {"load_in_4bit": True}
        elif self.load_bit == "fp16":
            precision = {"torch_dtype": torch.float16}
        elif self.load_bit == "bf16":
            precision = {"torch_dtype": torch.bfloat16}
        else:
            precision = {}
        if "otter" in checkpoint_path.lower():
            model = OtterForConditionalGeneration.from_pretrained(checkpoint_path, device_map={"": "cuda:0"}, **precision)
        else:
            model = FlamingoForConditionalGeneration.from_pretrained(checkpoint_path, device_map=device_map, **precision)
        model.text_tokenizer.padding_side = "left"  # otter video
        tokenizer = model.text_tokenizer

        if num_gpus > 0:
            model.cuda()
        model.eval()
        model.tie_weights()

        self.device = "cuda" if num_gpus > 0 else "cpu"
        logger.info(f"Loading the model to {self.device} in {self.load_bit}...")
        context_len = 2048
        image_processor = transformers.CLIPImageProcessor()

        return tokenizer, model, image_processor, context_len

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {[self.model_name]}. " f"Semaphore: {pretty_print_semaphore(model_semaphore)}. " f"global_counter: {global_counter}"
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=25,
                )
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
            return args.limit_model_concurrency - model_semaphore._value + (len(model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        logger.info(f"Generate stream...")
        tokenizer, model, image_processor = (
            self.tokenizer,
            self.model,
            self.image_processor,
        )
        prompt = params["prompt"]
        logger.info(f"Prompt:::{prompt}")
        images = params.get("images", None)

        if images is not None:
            assert type(images) is list
            if len(images) > 0:
                if type(images[0]) is list:  # current support single video
                    images = images[-1]
                    is_video = True
                else:
                    is_video = False
                # cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
                images = [Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB") for image in images]
                logger.info(f"{len(images)} images conditioned.")
                tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[self.load_bit]
                if is_video is True:
                    vision_x = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
                    assert vision_x.shape[2] == len(images)  # dim of vision_x: [B, T, F, C, H, W], make sure conditioned on frames of the same video
                else:
                    vision_x = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
                vision_x = vision_x.to(self.device, dtype=tensor_dtype)
                logger.info(f"Is video? {is_video} vision_x shape: {vision_x.shape}")
            else:
                images = None
                vision_x = None

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        logger.info(f"Input prompt: {prompt}")
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
        ).to(self.device)
        logger.info(f"input_ids: {inputs['input_ids'].shape} attention_mask: {inputs['attention_mask'].shape}")
        generation_kwargs = params.get("generation_kwargs", {})
        # generation_kwargs["num_beams"] = generation_kwargs.get("num_beams", 3)
        logger.info(f"generation_kwargs: {generation_kwargs}")

        # vision_x = vision_x.to(self.model.device)
        lang_x = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        bad_words_id = tokenizer(["User:", "GPT:"], add_special_tokens=False).input_ids
        generation_input = dict(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            streamer=streamer,
            **generation_kwargs,
        )
        # # Call the generate function and store the output in a variable
        # generated_output = model.generate(**generation_input)

        # # Decode the output using the tokenizer
        # generated_text = (
        #     tokenizer.decode(generated_output[0])
        #     .split("<answer>")[-1]
        #     .lstrip()
        #     .rstrip()
        #     .split("<|endofchunk|>")[0]
        #     .lstrip()
        #     .rstrip()
        #     .lstrip('"')
        #     .rstrip('"')
        # )
        # logger.info(f"Generated text: {generated_text}")
        # ret = {
        #     "text": generated_text,
        #     "error_code": 0,
        # }
        # yield json.dumps(ret).encode() + b"\0"
        thread = threading.Thread(target=model.generate, kwargs=generation_input)
        thread.start()
        generated_text = ""
        for i, output in enumerate(streamer):
            generated_text += output
            if "IMPRESSION" in generated_text:
                generated_text = generated_text.replace("IMPRESSION", "\nIMPRESSION")
            if i % 10 == 0:
                logger.info(f"Generated text: {generated_text}")
            ret = {
                "text": generated_text,
                "error_code": 0,
            }
            yield json.dumps(ret).encode() + b"\0"

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
    parser.add_argument("--worker_address", type=str, default="http://localhost:21002")
    parser.add_argument("--controller_address", type=str, default="http://localhost:21001")
    parser.add_argument("--lm_path", type=str, default="luodian/llama-7b-hf")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--keep_aspect_ratio", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--limit_model_concurrency", type=int, default=5)
    parser.add_argument("--stream_interval", type=int, default=2)
    parser.add_argument("--no_register", action="store_true")
    parser.add_argument("--load_bit", type=str, choices=["fp16", "bf16", "int8", "int4", "fp32"], default="fp32")
    parser.add_argument("--load_pt", action="store_true")
    args = parser.parse_args()

    worker_id = str(uuid.uuid4())[:6]
    logger = build_logger("model_worker", f"model_worker_{args.model_name}_{worker_id}.log")

    logger.info(f"args: {args}")

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.lm_path,
        args.model_name,
        args.checkpoint_path,
        args.keep_aspect_ratio,
        args.num_gpus,
        args.load_bit,
        args.load_pt,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
