import argparse

import torch
from torch import nn

from ofa.configuration_ofa import OFAConfig
from ofa.modeling_ofa import OFAModel
from architecture_configs import ofa_base, ofa_large, ofa_tiny


def trans_fairseq_to_huggingface(fs_model, hf_model, config):
    model = torch.load(fs_model, map_location='cpu')
    state = model["model"]
    keys = list(state.keys())
    for k in keys:
        if 'version' in k:
            del state[k]
            continue
        new_k = k.replace('self_attn_ln', 'self_attn_mid_layer_norm').\
                  replace('ffn_layernorm', 'ffn_layer_norm').\
                  replace('cross_attn_ln', 'cross_attn_mid_layer_norm').\
                  replace('encoder_attn', 'cross_attn').\
                  replace('attn_ln', 'self_attn_mid_layer_norm')
        v = state[k]
        del state[k]
        state[new_k] = v
    model["model"] = state
    remove_ignore_keys_(state)
    ofa_config = OFAConfig(**config)
    model = OFAModel(ofa_config)
    model.load_state_dict(state)
    model.save_pretrained(hf_model)


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert ofa original ckpt to huggingface.')
    parser.add_argument('--pt_model', type=str, default='',
                        help='path of original ckpt')
    parser.add_argument('--hf_model_dir', type=str, default='',
                        help='directory of huggingface ckpt')
    args = parser.parse_args()
    trans_fairseq_to_huggingface(args.pt_model, args.hf_model_dir, ofa_large)
