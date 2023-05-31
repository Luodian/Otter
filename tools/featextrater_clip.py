"""
Extract features using CLIP.
"""
import os
import sys
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torchvision.models as models
from torchvision import transforms as T
from torch.nn import functional as F


import timm

sys.path.append("/mnt/lustre/yhzhang/OFA-Compress")
from data_utils.input_dataset import FileDataset
from io import BytesIO
import base64
import json
from datetime import datetime, timedelta
import clip

import re


def initialize_distributed():
    """Initialize torch.distributed."""
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    # Manually set the device ids.
    device = rank % torch.cuda.device_count()
    print("device id: {}".format(device))
    torch.cuda.set_device(device)
    # Call the init process
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    init_method += master_ip + ":" + master_port
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        init_method=init_method,
        timeout=timedelta(seconds=3000),
    )
    print("world_size =", world_size, ", rank =", rank)
    assert rank == torch.distributed.get_rank()


def pre_question(question, max_ques_words):
    question = question.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ")

    question = re.sub(
        r"\s{2,}",
        " ",
        question,
    )
    question = question.rstrip("\n")
    question = question.strip(" ")

    # truncate question
    question_words = question.split(" ")
    if len(question_words) > max_ques_words:
        question = " ".join(question_words[:max_ques_words])

    return question


def pre_answer(answer, max_ans_words):
    answer = re.sub(
        r"\s{2,}",
        " ",
        answer,
    )
    answer = answer.rstrip("\n")
    answer = answer.strip(" ")

    # truncate question
    return_answer = ""
    answers = answer.split(".")

    for _ in answers:
        if return_answer == "":
            cur_answer = _
        else:
            cur_answer = ".".join([return_answer, _])
        if len(cur_answer.split(" ")) <= max_ans_words:
            return_answer = cur_answer
        else:
            break

    if return_answer == "":
        answer_words = answer.split(" ")
        return_answer = " ".join(answer_words[:max_ques_words])
    else:
        if return_answer[-1] != "." and return_answer != answers:
            return_answer += "."

    return return_answer


sys.path.append("/mnt/lustre/yhzhang/OFA-Compress/data_utils")
from transforms import ObjectCrop


initialize_distributed()
rank = torch.distributed.get_rank()

model_name = "ViT-B/16"
model, preprocess = clip.load(model_name, device="cuda")
model = model.to(rank)


model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[rank],
    output_device=rank,
    find_unused_parameters=True,
    broadcast_buffers=False,
)
model.eval()

root = f"/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/features/"

save_name = f"{root}/clip_vitb16_features"

dataset = FileDataset(
    "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/complex_reasoning_77k.tsv",
    "0,1,2,3,4,5,6,7",
)

global_features_img = torch.tensor([]).to(rank)
global_features_text = torch.tensor([]).to(rank)
global_features_texts_with_answer = torch.tensor([]).to(rank)
imgs = []
texts = []
texts_with_answer = []
uniq_ids = []

for cur_idx in tqdm(range(dataset.get_cur_slice_len())):
    _ = dataset.__getitem__(0)
    uniq_id, image, caption, question, refs, gt_objects, dataset_name, type = _
    uniq_ids.append(uniq_id)
    try:
        img = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
        if dataset_name == "complex_reasoning_77k":
            max_src_length = max_tgt_length = 256
            question = pre_question(question, max_src_length)
            question = question.strip("<image>")
            answer = refs.strip().replace("#", " ")
            answer = pre_answer(answer, max_tgt_length)
        img = preprocess(img).to(rank)
        text = clip.tokenize([question]).squeeze().to(rank)
        imgs.append(img)
        texts.append(text)
    except:
        print(f"Disappear {path}")
        sys.stdout.flush()

    if len(imgs) == 128:
        # import pdb;pdb.set_trace()
        imgs = torch.stack(imgs).to(rank)
        texts = torch.stack(texts).to(rank)
        with torch.no_grad():
            image_features = model.module.encode_image(imgs)
            # import pdb;pdb.set_trace()
            text_features = model.module.encode_text(texts)
            if len(global_features_img) == 0:
                global_features_img = image_features
                global_features_text = text_features
            else:
                global_features_img = torch.cat((global_features_img, image_features))
                global_features_text = torch.cat((global_features_text, text_features))

        imgs = []
        texts = []


imgs = torch.stack(imgs).to(rank)
texts = torch.stack(texts).to(rank)

with torch.no_grad():
    image_features = model.module.encode_image(imgs)
    text_features = model.module.encode_text(texts)

    if len(global_features_img) == 0:
        global_features_img = image_features
        global_features_text = text_features
    else:
        global_features_img = torch.cat((global_features_img, image_features))
        global_features_text = torch.cat((global_features_text, text_features))
# import pdb;pdb.set_trace()
imgs_features = global_features_img.cpu().numpy().astype(np.float32)
text_features = global_features_text.cpu().numpy().astype(np.float32)

np.savez(
    f"{save_name}.rank_{rank}",
    uniqids=uniq_ids,
    image_features=imgs_features,
    text_features=text_features,
)
