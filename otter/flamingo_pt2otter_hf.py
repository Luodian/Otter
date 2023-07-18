""" convert from open flamingo pt to otter hf, as the starting point for ICI training
    Deprecated and should change for MPT version OtterModel
"""

import re
import argparse
import os

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

from otter.modeling_otter import (
    OtterPreTrainedModel,
    OtterLMMixin,
    extend_instance,
    _infer_decoder_layers_attr_name,
    OtterPerceiverResampler,
)

from otter.configuration_otter import OtterConfig


class OtterModel(OtterPreTrainedModel):
    # We need to download the llaMA and CLIP here, and the model does not have the <answer> when init
    config_class = OtterConfig

    def __init__(
        self,
        config: OtterConfig,
    ):
        super().__init__(config)
        text_tokenizer = LlamaTokenizer.from_pretrained(config.text_config._name_or_path)
        lang_encoder = LlamaForCausalLM.from_pretrained(config.text_config._name_or_path)
        vision_encoder = CLIPVisionModel.from_pretrained(config.vision_config._name_or_path)

        text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.text_tokenizer = text_tokenizer
        self.eoc_token_id = text_tokenizer.encode("<|endofchunk|>")[-1]
        self.media_token_id = text_tokenizer.encode("<image>")[-1]

        extend_instance(lang_encoder, OtterLMMixin)
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        lang_encoder.resize_token_embeddings(len(text_tokenizer))
        self.lang_encoder = lang_encoder

        self.cross_attn_every_n_layers = config.cross_attn_every_n_layers
        self.use_media_placement_augmentation = config.use_media_placement_augmentation
        self.only_attend_previous = config.only_attend_previous

        vision_encoder.output_tokens = True
        self.vision_encoder = vision_encoder

        self.vis_dim = 1024
        self.perceiver = OtterPerceiverResampler(dim=self.vis_dim)

        self.lang_encoder.init_otter(
            media_token_id=self.media_token_id,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=self.cross_attn_every_n_layers,
            use_media_placement_augmentation=self.use_media_placement_augmentation,
            only_attend_previous=self.only_attend_previous,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.lang_encoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.lang_encoder.set_output_embeddings(new_embeddings)


def rename_flamingo_checkpoint(old_ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename some keys in the public flamingo checkpoint"""
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


@torch.no_grad()
def dump_hf_model(old_ckpt_path: str, new_folder_path: str) -> None:
    os.makedirs(new_folder_path, exist_ok=True)
    old_ckpt = torch.load(old_ckpt_path, map_location="cpu")
    if old_ckpt.get("model", None) is not None:
        old_ckpt = old_ckpt["model"]
    config = OtterConfig.from_json_file("otter/config.json")
    model = OtterModel(config)
    new_ckpt = rename_flamingo_checkpoint(old_ckpt)
    model.load_state_dict(new_ckpt, strict=False)
    text_tokenizer = model.text_tokenizer
    text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]})
    model.lang_encoder.resize_token_embeddings(len(text_tokenizer))
    print(f"Saving HF model to {new_folder_path}")
    model.save_pretrained(new_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old_ckpt_path",
        "-old",
        type=str,
        required=True,
        help="Path to the Open Flamingo checkpoint",
    )
    parser.add_argument(
        "--new_hf_path",
        "-new",
        type=str,
        required=True,
        help="Path to the HF folder",
    )
    args = parser.parse_args()
    dump_hf_model(args.old_ckpt_path, args.new_hf_path)
