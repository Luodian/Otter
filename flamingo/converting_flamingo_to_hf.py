import argparse
import os
import re

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer
from transformers.models.auto import AutoTokenizer

from flamingo.configuration_flamingo import FlamingoConfig
from flamingo.falcon.modelling_RW import RWForCausalLM
from flamingo.modeling_flamingo import (
    FlamingoConfig,
    FlamingoLMMixin,
    FlamingoPerceiverResampler,
    FlamingoPreTrainedModel,
    _infer_decoder_layers_attr_name,
    extend_instance,
)
from flamingo.mpt.modeling_mpt import MPTForCausalLM

from .configuration_flamingo import FlamingoConfig


class FlamingoModel(FlamingoPreTrainedModel):
    config_class = FlamingoConfig

    def __init__(
        self,
        config: FlamingoConfig,
        args,
    ):
        super().__init__(config)
        ### TODO: give "LlamaForCausalLM" as the name of text_config.architectures of Llama_based flamingo
        if "llama" not in config.text_config._name_or_path:
            if config.text_config.architectures[0] == "MPTForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b-instruct")
                lang_encoder = MPTForCausalLM(config=config.text_config)
            elif config.text_config.architectures[0] == "RWForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained("PATH-TO-YOUR-FALCON")
                lang_encoder = RWForCausalLM(config=config.text_config)
        else:
            text_tokenizer = LlamaTokenizer.from_pretrained(config.text_config._name_or_path)
            lang_encoder = LlamaForCausalLM(config=config.text_config)

        vision_encoder = CLIPVisionModel(config=config.vision_config)
        text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.text_tokenizer = text_tokenizer
        self.eoc_token_id = text_tokenizer.encode("<|endofchunk|>")[-1]
        self.media_token_id = text_tokenizer.encode("<image>")[-1]

        extend_instance(lang_encoder, FlamingoLMMixin)
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        lang_encoder.resize_token_embeddings(len(text_tokenizer))
        self.lang_encoder = lang_encoder

        self.cross_attn_every_n_layers = config.cross_attn_every_n_layers
        self.use_media_placement_augmentation = config.use_media_placement_augmentation

        vision_encoder.output_tokens = True
        self.vision_encoder = vision_encoder

        self.vis_dim = 1024
        self.perceiver = FlamingoPerceiverResampler(dim=self.vis_dim)

        self.lang_encoder.init_flamingo(
            media_token_id=self.media_token_id,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=self.cross_attn_every_n_layers,
            use_media_placement_augmentation=self.use_media_placement_augmentation,
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
    """Rename some keys in the public Flamingo checkpoint"""
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
def dump_hf_model(old_ckpt_path: str, new_folder_path: str, args) -> None:
    old_ckpt = torch.load(old_ckpt_path, map_location="cpu")
    if old_ckpt.get("model", None) is not None:
        old_ckpt = old_ckpt["model"]

    old_folder_path = os.path.dirname(old_ckpt_path)
    config_file = args.config_file if args.config_file else os.path.join(old_folder_path, "config.json")
    config = FlamingoConfig.from_json_file(config_file)
    print("Initializing HF model")
    model = FlamingoModel(config, args)
    new_ckpt = rename_flamingo_checkpoint(old_ckpt)
    model.load_state_dict(new_ckpt, strict=False)
    print(f"Saving HF model to {new_folder_path}")
    model.save_pretrained(new_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old_ckpt_path",
        "-old",
        type=str,
        required=True,
        help="Path to the OpenFlamingo checkpoint",
    )
    parser.add_argument("--add-answer-token", action="store_true")
    parser.add_argument(
        "--new_hf_path",
        "-new",
        type=str,
        required=True,
        help="Path to the HF folder",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to a HF config file",
    )
    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.new_hf_path)):
        os.makedirs(os.path.dirname(args.new_hf_path))
    dump_hf_model(args.old_ckpt_path, args.new_hf_path, args)
