import argparse
import json
from math import ceil
import os
import random
import uuid
from collections import defaultdict
from typing import Callable

import more_itertools
import numpy as np
import torch
from open_flamingo_archived.eval.coco_metric import compute_cider, postprocess_captioning_generation
from open_flamingo_archived.eval.eval_datasets import COCOFlickrDataset, VQADataset, ImageNetDataset
from tqdm import tqdm

from open_flamingo_archived.eval.ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo_archived.eval.vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation
from open_flamingo_archived.eval.classification import (
    compute_per_sample_probs,
    compute_per_sample_loss,
)

from open_flamingo_archived.train.train_utils import get_autocast, get_cast_dtype
from open_flamingo_archived.eval.imagenet_utils import (
    openai_imagenet_classnames,
    IMAGENET_1K_CLASS_ID_TO_LABEL,
)

from open_flamingo_archived.src.factory import create_model_and_transforms

import nltk

nltk.download("averaged_perceptron_tagger", quiet=True)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
parser.add_argument("--lm_tokenizer_path", type=str, default="facebook/opt-30b")
parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument(
    "--cross_attn_every_n_layers",
    type=int,
    default=1,
    help="how often to add a cross-attention layer after each transformer layer",
)
parser.add_argument("--results_file", type=str, default=None, help="JSON file to save results")

# Trial arguments
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    default=[0],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples to evaluate on")

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--device", type=int, default=0)

parser.add_argument(
    "--eval_ok_vqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on OK-VQA.",
)

# Dataset arguments
## OK-VQA Dataset
parser.add_argument(
    "--ok_vqa_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
    default=None,
)
parser.add_argument("--weight_decay", default=0.1, type=float)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument(
    "--precision",
    choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
    default="fp32",
    help="Floating point precision.",
)


def main():
    args = parser.parse_args()

    # load model
    flamingo, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.lm_tokenizer_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    flamingo.load_state_dict(checkpoint, strict=False)
    flamingo.to(args.device if args.device >= 0 else "cpu")

    print("Training on OK-VQA...")
    train_one_epoch(
        model=flamingo,
        tokenizer=tokenizer,
        image_processor=image_processor,
        batch_size=args.batch_size,
        device=args.device,
        image_dir_path=args.ok_vqa_image_dir_path,
        questions_json_path=args.ok_vqa_questions_json_path,
        annotations_json_path=args.ok_vqa_annotations_json_path,
        vqa_dataset="ok_vqa",
        args=args,
    )
    pass


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(f"num_samples + num_shots must be less than {len(full_dataset)}")

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(len(full_dataset), num_samples + query_set_size, replace=False)
    return random_indices


def get_outputs(
    model,
    batch_images,
    device,
    attention_mask,
    max_generation_length,
    num_beams,
    length_penalty,
    input_ids,
):
    with torch.inference_mode():
        outputs = model.generate(
            batch_images.to(device if device >= 0 else "cpu"),
            input_ids.to(device if device >= 0 else "cpu"),
            attention_mask=attention_mask.to(device if device >= 0 else "cpu"),
            max_new_tokens=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

    outputs = outputs[:, len(input_ids[0]) :]
    return outputs


def train_one_epoch(model, tokenizer, image_processor, batch_size, image_dir_path, questions_json_path, annotations_json_path, seed=42, device=-1, vqa_dataset="vqa", args=None):
    full_dataset = VQADataset(
        image_dir_path=image_dir_path,
        question_path=questions_json_path,
        annotations_path=annotations_json_path,
        vqa_dataset=vqa_dataset,
    )
    max_generation_length = 5
    num_beams = 3
    length_penalty = -2.0

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return "gated_cross_attn_layer" in x and "ff_gate" not in x and "attn_gate" not in x and "norm" not in x and "bias" not in x

        for n, p in model.named_parameters():
            # if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(get_grouped_params(model), lr=args.learning_rate)
    # total_training_steps = (len(full_dataset) // (batch_size * args.world_size)) * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    for batch in more_itertools.chunked(tqdm(full_dataset, desc="Running inference"), batch_size):
        device = torch.device("cuda" if device >= 0 else "cpu")
        batch_images = [image_processor(s["image"]).unsqueeze(0) for s in batch]
        batch_texts = [s["question"] for s in batch]
        batch_labels = [s["answers"][0] for s in batch]

        images = torch.cat(batch_images, dim=0).to(device, dtype=cast_dtype, non_blocking=True)

        tokenizer.padding_side = "left"
        encodings = tokenizer(batch_texts, return_tensors="pt", padding="longest", truncation=True, max_length=2000)
        input_ids = encodings["input_ids"].to(device, dtype=cast_dtype, non_blocking=True)
        attention_mask = encodings["attention_mask"].to(device, dtype=cast_dtype, non_blocking=True)
        labels = tokenizer(batch_labels, return_tensors="pt", padding="longest", truncation=True)["input_ids"].to(device, dtype=cast_dtype, non_blocking=True)

        with autocast():
            loss = model(
                vision_x=images.unsqueeze(1).unsqueeze(2),
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

        # process_function = postprocess_vqa_generation if vqa_dataset == "vqa" else postprocess_ok_vqa_generation

        # new_predictions = [process_function(out) for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)]

if __name__ == "__main__":
    main()
