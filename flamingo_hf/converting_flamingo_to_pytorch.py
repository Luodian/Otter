import re
import argparse
import os
import torch


@torch.no_grad()
def convert_flamingo_checkpoint(
    old_ckpt: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Convert the public Flamingo checkpoint to the checkpoint we need."""
    perceiver_pattern1 = re.compile(r"perceiver\.layers\.[0-9]\.0")
    perceiver_pattern2 = re.compile(r"perceiver\.layers\.[0-9]\.1")
    new_ckpt = old_ckpt.copy()
    for key, value in old_ckpt.items():
        if re.match(perceiver_pattern1, key):
            new_key = re.sub(r"([0-9])\.0", r"\1", key)
            new_ckpt.pop(key)
            new_ckpt[new_key] = value
        elif re.match(perceiver_pattern2, key):
            new_key = re.sub(r"([0-9])\.1", r"\1.feed_forward", key)
            new_ckpt.pop(key)
            new_ckpt[new_key] = value
        elif key.startswith("lang_encoder.gated_cross_attn_layers."):
            new_ckpt.pop(key)
        elif key.startswith("lang_encoder.") and "ff_gate" not in key:
            new_key = key.replace("ff", "feed_forward")
            new_ckpt.pop(key)
            new_ckpt[new_key] = value
        
    return new_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old", type=str, required=True, default="flamingo_hf/checkpoint.pt"
    )
    parser.add_argument(
        "--new", type=str, required=True, default="flamingo_hf/hf_checkpoint.pt"
    )
    args = parser.parse_args()
    old_ckpt = torch.load(args.old, map_location="cpu")
    new_ckpt = convert_flamingo_checkpoint(old_ckpt)
    if not os.path.exists(os.path.dirname(args.new)):
        os.makedirs(os.path.dirname(args.new))
    torch.save(new_ckpt, args.new)
