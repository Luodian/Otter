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
from open_flamingo_archived.eval.imagenet_utils import (
    openai_imagenet_classnames,
    IMAGENET_1K_CLASS_ID_TO_LABEL,
)

from open_flamingo_archived.src.factory import create_model_and_transforms

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
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
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
parser.add_argument(
    "--num_samples", type=int, default=5000, help="Number of samples to evaluate on"
)

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--device", type=int, default=0)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_coco",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)
parser.add_argument(
    "--eval_vqav2",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQAV2.",
)
parser.add_argument(
    "--eval_ok_vqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on OK-VQA.",
)
parser.add_argument(
    "--eval_imagenet",
    action="store_true",
    default=False,
    help="Whether to evaluate on ImageNet.",
)

parser.add_argument(
    "--eval_flickr30",
    action="store_true",
    default=False,
    help="Whether to evaluate on Flickr30.",
)

# Dataset arguments

## Flickr30 Dataset
parser.add_argument(
    "--flickr_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
    default=None,
)

## COCO Dataset
parser.add_argument(
    "--coco_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default=None,
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_annotations_json_path",
    type=str,
    default=None,
)

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

## Imagenet dataset
parser.add_argument("--imagenet_root", type=str, default="/tmp")


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

    results = defaultdict(list)
    if args.eval_vqav2:
        print("Evaluating on VQAv2...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    model=flamingo,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    image_dir_path=args.vqav2_image_dir_path,
                    questions_json_path=args.vqav2_questions_json_path,
                    annotations_json_path=args.vqav2_annotations_json_path,
                    vqa_dataset="vqa",
                )
                print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                scores.append(vqa_score)
            print(f"Shots {shot} Mean VQA score: {np.mean(scores)}")
            results["vqav2"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.eval_imagenet:
        print("Evaluating on ImageNet...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                imagenet_score = evaluate_imagenet(
                    model=flamingo,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    imagenet_root=args.imagenet_root,
                )
                print(
                    f"Shots {shot} Trial {trial} " f"ImageNet score: {imagenet_score}"
                )
                scores.append(imagenet_score)
            print(f"Shots {shot} Mean ImageNet score: {np.mean(scores)}")
            results["imagenet"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def prepare_eval_samples_and_dataset(full_dataset, random_indices, query_set_size):
    # get in context samples
    in_context_samples = [full_dataset[i] for i in random_indices[:query_set_size]]
    eval_dataset = torch.utils.data.Subset(
        full_dataset, random_indices[query_set_size:]
    )
    return in_context_samples, eval_dataset


def get_context_images(image_processor, in_context_samples, num_shots):
    if num_shots > 0:
        context_images = [
            image_processor(s["image"]).unsqueeze(0) for s in in_context_samples
        ]
        context_images = torch.cat(context_images, dim=0)
        context_images = context_images.unsqueeze(1).unsqueeze(0)
    else:
        context_images = None
    return context_images


def get_context_text(
    get_prompt: Callable[[dict], str],
    in_context_samples,
    effective_num_shots,
    num_shots,
) -> str:
    context_text = (
        "".join([get_prompt(s) for s in in_context_samples])
        if effective_num_shots > 0
        else ""
    )

    if num_shots == 0:
        context_text = context_text.replace("<image>", "")
    return context_text


def prepare_batch_images(batch, image_processor, context_images, num_shots):
    batch_images = None
    for b, sample_imgs in zip(batch, context_images):
        b_image = image_processor(b["image"]).unsqueeze(0).unsqueeze(1).unsqueeze(0)
        b_image = torch.cat([sample_imgs, b_image], dim=1) if num_shots > 0 else b_image

        if batch_images is None:
            batch_images = b_image
        else:
            batch_images = torch.cat([batch_images, b_image], dim=0)
    return batch_images


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


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

def evaluate_vqa(
    model,
    tokenizer,
    image_processor,
    batch_size,
    image_dir_path,
    questions_json_path,
    annotations_json_path,
    seed=42,
    max_generation_length=5,
    num_beams=3,
    length_penalty=-2.0,
    num_samples=5000,
    query_set_size=2048,
    num_shots=8,
    device=-1,
    vqa_dataset="vqa",
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor : image processor for the model
        batch_size (int): batch size
        image_dir_path (str): path to image directory
        questions_json_path (str): path to questions json file
        annotations_json_path (str): path to annotations json file
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        query_set_size (int, optional): size of the query set. Defaults to 2048.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1 (cpu).
        num_workers (int, optional): number of workers to use. Defaults to 4.
        vqa_dataset (string): type of vqa dataset: currently supports vqa, ok_vqa. Defaults to vqa.
    Returns:
        float: accuracy score
    """

    full_dataset = VQADataset(
        image_dir_path=image_dir_path,
        question_path=questions_json_path,
        annotations_path=annotations_json_path,
        vqa_dataset=vqa_dataset,
    )

    effective_num_shots = num_shots if num_shots > 0 else 2

    if num_samples + effective_num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than or equal to {len(full_dataset)}"
        )

    random_indices = get_random_indices(num_samples, query_set_size, full_dataset, seed)

    def get_prompt(sample, train=True):
        return f"<image>Question:{sample['question'].strip()} Short Answer:{sample['answers'][0].strip() if train else ''}{'<|endofchunk|>' if train else ''}"

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset,
        random_indices=random_indices,
        query_set_size=query_set_size,
    )

    model.eval()
    predictions = []

    for batch in more_itertools.chunked(
        tqdm(eval_dataset, desc="Running inference"), batch_size
    ):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch)
        )

        context_images = [
            get_context_images(
                image_processor=image_processor,
                in_context_samples=batch_demo_samples[i],
                num_shots=num_shots,
            )
            for i in range(len(batch))
        ]

        context_text = [
            get_context_text(
                get_prompt,
                in_context_samples=batch_demo_samples[i],
                effective_num_shots=effective_num_shots,
                num_shots=num_shots,
            )
            for i in range(len(batch))
        ]

        batch_images = prepare_batch_images(
            batch=batch,
            image_processor=image_processor,
            context_images=context_images,
            num_shots=num_shots,
        )

        batch_text = [
            context_text[i] + get_prompt(s, train=False) for i, s in enumerate(batch)
        ]

        tokenizer.padding_side = "left"
        encodings = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=2000,
        )
        input_ids = encodings["input_ids"].to(device if device >= 0 else "cpu")
        attention_mask = encodings["attention_mask"].to(
            device if device >= 0 else "cpu"
        )

        outputs = get_outputs(
            model=model,
            batch_images=batch_images,
            device=device,
            attention_mask=attention_mask,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            input_ids=input_ids,
        )

        process_function = (
            postprocess_vqa_generation
            if vqa_dataset == "vqa"
            else postprocess_ok_vqa_generation
        )

        new_predictions = [
            process_function(out)
            for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        predictions.extend(
            [
                {"answer": p, "question_id": sample["question_id"]}
                for p, sample in zip(new_predictions, batch)
            ]
        )
    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"{vqa_dataset}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f"{vqa_dataset}results_{random_uuid}.json",
        questions_json_path,
        annotations_json_path,
    )

    # delete the temporary file
    os.remove(f"{vqa_dataset}results_{random_uuid}.json")

    return acc

if __name__ == "__main__":
    main()
