"""convert from otter pt to otter hf. Will remove after we use otter hf model to train.
"""

import re
import argparse
import os

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

import sys
from modeling_flamingo import FlamingoForConditionalGeneration

from configuration_flamingo import FlamingoConfig


@torch.no_grad()
def dump_hf_model(pretrained_model_path: str, old_ckpt_path: str, new_folder_path: str) -> None:
    old_ckpt = torch.load(old_ckpt_path, map_location="cpu")
    if old_ckpt.get("model_state_dict", None) is not None:
        old_ckpt = old_ckpt["model_state_dict"]
    new_ckpt = old_ckpt
    folder_path = os.path.dirname(old_ckpt_path)
    # config_path = os.path.join(folder_path, "config.json") if os.path.exists(os.path.join(folder_path, "config.json")) else "flamingo/config.json"
    model = FlamingoForConditionalGeneration.from_pretrained(
        args.pretrained_model_path,
        device_map="auto",
    )
    _ = model.load_state_dict(new_ckpt, strict=False)
    print(f"Saving HF model to {new_folder_path}")
    model.save_pretrained(new_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old_ckpt_path",
        "-old",
        type=str,
        required=True,
        help="Path to the pt checkpoint",
    )
    parser.add_argument(
        "--new_hf_path",
        "-new",
        type=str,
        required=True,
        help="Path to the hf folder",
    )
    parser.add_argument(
        "--pretrained_model_path",
        "-pretrained",
        type=str,
        required=True,
        help="Path to the pretrained model folder",
    )
    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.new_hf_path)):
        os.makedirs(os.path.dirname(args.new_hf_path))
    dump_hf_model(args.pretrained_model_path, args.old_ckpt_path, args.new_hf_path)
