import re
import argparse
import os

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

from modeling_otter import (
    OtterConfig,
    OtterPreTrainedModel,
    OtterLMMixin,
    extend_instance,
    _infer_decoder_layers_attr_name,
    OtterPerceiverResampler,
)

from configuration_otter import OtterConfig


class OtterModel(OtterPreTrainedModel):
    config_class = OtterConfig

    def __init__(
        self,
        config: OtterConfig,
    ):
        super().__init__(config)
        text_tokenizer = LlamaTokenizer.from_pretrained(
            config.text_config._name_or_path
        )
        lang_encoder = LlamaForCausalLM.from_pretrained(
            config.text_config._name_or_path
        )
        vision_encoder = CLIPVisionModel.from_pretrained(
            config.vision_config._name_or_path
        )

        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]}
        )
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

        vision_encoder.output_tokens = True
        self.vision_encoder = vision_encoder

        self.vis_dim = 1024
        self.perceiver = OtterPerceiverResampler(dim=self.vis_dim)

        self.lang_encoder.init_otter(
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


@torch.no_grad()
def dump_hf_model(old_ckpt_path: str, new_folder_path: str) -> None:
    old_ckpt = torch.load(old_ckpt_path, map_location="cpu")
    if old_ckpt.get("model", None) is not None:
        old_ckpt = old_ckpt["model"]
    config = OtterConfig.from_json_file("otter_hf/config.json")
    model = OtterModel(config)
    model.load_state_dict(old_ckpt, strict=False)
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
    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.new_hf_path)):
        os.makedirs(os.path.dirname(args.new_hf_path))
    dump_hf_model(args.old_ckpt_path, args.new_hf_path)
