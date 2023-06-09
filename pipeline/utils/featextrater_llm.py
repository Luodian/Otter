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

import re
from transformers import AutoTokenizer, AutoModel


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


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


sys.path.append("/mnt/lustre/yhzhang/OFA-Compress/data_utils")
from transforms import ObjectCrop


initialize_distributed()
rank = torch.distributed.get_rank()

llm_version = "all-MiniLM-L6-v1"
tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{llm_version}")
model = AutoModel.from_pretrained(f"sentence-transformers/{llm_version}")

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

save_name = f"{root}/{llm_version}_features"

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
        if dataset_name == "complex_reasoning_77k":
            max_src_length = max_tgt_length = 256
            question = pre_question(question, max_src_length)
            question = question.strip("<image>")
            answer = refs.strip().replace("#", " ")
            answer = pre_answer(answer, max_tgt_length)
        texts.append(question)
    except:
        print(f"Disappear {path}")
        sys.stdout.flush()

    if len(texts) == 128:
        # import pdb;pdb.set_trace()
        encodings = tokenizer(
            texts,  # the texts to be tokenized
            padding=True,  # pad the texts to the maximum length (so that all outputs have the same length)
            return_tensors="pt",  # return the tensors (not lists)
        ).to(rank)
        with torch.no_grad():
            # text_features = model.module(**encodings)["pooler_output"]
            model_output = model.module(**encodings)
            text_features = mean_pooling(model_output, encodings["attention_mask"])
            if len(global_features_text) == 0:
                global_features_text = text_features
            else:
                global_features_text = torch.cat((global_features_text, text_features))
        # import pdb;pdb.set_trace()
        texts = []

encodings = tokenizer(
    texts,  # the texts to be tokenized
    padding=True,  # pad the texts to the maximum length (so that all outputs have the same length)
    return_tensors="pt",  # return the tensors (not lists)
).to(rank)
with torch.no_grad():
    # text_features = model.module(**encodings)["pooler_output"]
    model_output = model.module(**encodings)
    text_features = mean_pooling(model_output, encodings["attention_mask"])
    if len(global_features_text) == 0:
        global_features_text = text_features
    else:
        global_features_text = torch.cat((global_features_text, text_features))
# import pdb;pdb.set_trace()
text_features = global_features_text.cpu().numpy().astype(np.float32)
# import pdb;pdb.set_trace()
np.savez(
    f"{save_name}.rank_{rank}", uniqids=uniq_ids, text_features=text_features
)  # text_feafures for "coco_clip_vitb16_caption_test_features.rank_1.npz"
