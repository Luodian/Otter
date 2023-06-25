import argparse
import os

import torch

from .configuration_flamingo import FlamingoConfig
from .modeling_flamingo import FlamingoForConditionalGeneration

parser = argparse.ArgumentParser(description="Load model with precision")
parser.add_argument("--load_bit", type=str, choices=["fp16", "bf16"], required=True, help="Choose either 'fp16' or 'bf16'")
args = parser.parse_args()

load_bit = args.load_bit

if load_bit == "fp16":
    precision = {"torch_dtype": torch.float16}
elif load_bit == "bf16":
    precision = {"torch_dtype": torch.bfloat16}

root_dir = os.environ["AZP"]
print(root_dir)
checkpoint_path = f"{root_dir}/otter/checkpoints/flamingo-mpt-7B-instruct-init"
device_id = "cpu"
model = FlamingoForConditionalGeneration.from_pretrained(checkpoint_path, device_map={"": device_id}, **precision)

# save model
checkpoint_path = checkpoint_path + f"_{load_bit}"
model.save_pretrained(checkpoint_path)
