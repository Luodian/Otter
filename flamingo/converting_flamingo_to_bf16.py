from .configuration_flamingo import FlamingoConfig
from .modeling_flamingo import FlamingoForConditionalGeneration
import torch
import os

load_bit = "bf16"

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
