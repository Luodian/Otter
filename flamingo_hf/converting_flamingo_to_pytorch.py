import re
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
        elif key.startswith("lang_encoder.") and "ff_gate" not in key:
            new_key = key.replace("ff", "feed_forward")
            new_ckpt.pop(key)
            new_ckpt[new_key] = value
    return new_ckpt


if __name__ == "__main__":
    old_ckpt = torch.load("flamingo_hf/checkpoint.pt")
    new_ckpt = convert_flamingo_checkpoint(old_ckpt)
    torch.save(new_ckpt, "flamingo_hf/hf_checkpoint.pt")
