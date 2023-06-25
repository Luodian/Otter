import argparse
import os

import torch
from tqdm import tqdm

from .configuration_flamingo import FlamingoConfig
from .modeling_flamingo import FlamingoForConditionalGeneration

parser = argparse.ArgumentParser(description="Convert MPT model")
parser.add_argument("--model_choice", type=str, choices=["7B", "30B"], required=True, help="Choose either '7B' or '30B'")
parser.add_argument("--save_root_dir", type=str, default="/home/luodian/projects/checkpoints", required=True)
args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
root_dir = os.environ["AZP"]
print(root_dir)

model_choice = args.model_choice
save_root_dir = args.save_root_dir

# prepare mpt model at first
# you can visit https://huggingface.co/mosaicml to download 7B and 30B instruct checkpoints.
if model_choice == "30B":
    config_file = "./flamingo/flamingo-mpt-30B.json"
    state_dict_files = [
        f"{root_dir}/otter/checkpoints/mpt-30b-instruct/pytorch_model-00001-of-00007.bin",
        f"{root_dir}/otter/checkpoints/mpt-30b-instruct/pytorch_model-00002-of-00007.bin",
        f"{root_dir}/otter/checkpoints/mpt-30b-instruct/pytorch_model-00003-of-00007.bin",
        f"{root_dir}/otter/checkpoints/mpt-30b-instruct/pytorch_model-00004-of-00007.bin",
        f"{root_dir}/otter/checkpoints/mpt-30b-instruct/pytorch_model-00005-of-00007.bin",
        f"{root_dir}/otter/checkpoints/mpt-30b-instruct/pytorch_model-00006-of-00007.bin",
        f"{root_dir}/otter/checkpoints/mpt-30b-instruct/pytorch_model-00007-of-00007.bin",
    ]
    save_path = f"{save_root_dir}/flamingo-mpt-30B-instruct-init"
elif model_choice == "7B":
    config_file = "./flamingo/flamingo-mpt-7B.json"
    state_dict_files = [
        f"{root_dir}/otter/checkpoints/mpt-7b-instruct/pytorch_model-00001-of-00002.bin",
        f"{root_dir}/otter/checkpoints/mpt-7b-instruct/pytorch_model-00002-of-00002.bin",
    ]
    save_path = f"{save_root_dir}/flamingo-mpt-7B-instruct-init"
else:
    raise ValueError("Invalid model_choice. Choose either '30B' or '7B'.")

config = FlamingoConfig.from_json_file(config_file)
model = FlamingoForConditionalGeneration(config=config)

state_dict = {}
for file in tqdm(state_dict_files, desc="Loading state dict"):
    state_dict_part = torch.load(file, map_location="cpu")
    state_dict.update(state_dict_part)

# load flamingo's vision encoder from last checkpoint.
# you can visit https://huggingface.co/luodian/openflamingo-9b-hf/tree/main to download the checkpoint.
state_dict_3 = torch.load(f"{root_dir}/otter/checkpoints/flamingo_9b_hf/pytorch_model-00004-of-00004.bin", map_location="cpu")
for cur_key in list(state_dict_3.keys()):
    if "vision_encoder" not in cur_key:
        del state_dict_3[cur_key]

load_msg = model.load_state_dict(
    state_dict_3,
    False,
)
# print incompatible keys
print(load_msg[1])

save_state_dict_1 = {}
for key in state_dict:
    if ".blocks." in key:
        _, _, layer_num, *remain_names = key.split(".")
        target_key = f"transformer.blocks.{layer_num}.decoder_layer.{'.'.join(remain_names)}"
    else:
        target_key = key
    save_state_dict_1[f"{target_key}"] = state_dict[key]

load_msg = model.lang_encoder.load_state_dict(
    save_state_dict_1,
    False,
)
# print incompatible keys
print(load_msg[1])

print(f"Saving model to {save_path}...")
model.save_pretrained(save_path, max_shard_size="10GB")
