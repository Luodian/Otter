import argparse
import os

import torch

from .configuration_flamingo import FlamingoConfig
from .modeling_flamingo import FlamingoForConditionalGeneration

parser = argparse.ArgumentParser(description="Load model with precision")
parser.add_argument("--load_bit", type=str, choices=["fp16", "bf16"], required=True, help="Choose either 'fp16' or 'bf16'")
parser.add_argument("--pretrained_model_path", type=str, default="/home/luodian/projects/checkpoints/flamingo-mpt-7B-instruct-init", required=True)
parser.add_argument("--saved_model_path", type=str, default="/home/luodian/projects/checkpoints/flamingo-mpt-7B-instruct-init", required=True)
args = parser.parse_args()

load_bit = args.load_bit
pretrained_model_path = args.pretrained_model_path

if load_bit == "fp16":
    precision = {"torch_dtype": torch.float16}
elif load_bit == "bf16":
    precision = {"torch_dtype": torch.bfloat16}

root_dir = os.environ["AZP"]
print(root_dir)
device_id = "cpu"
model = FlamingoForConditionalGeneration.from_pretrained(pretrained_model_path, device_map={"": device_id}, **precision)

# save model to same folder
checkpoint_path = pretrained_model_path + f"-{load_bit}"
model.save_pretrained(checkpoint_path, max_shard_size="10GB")
