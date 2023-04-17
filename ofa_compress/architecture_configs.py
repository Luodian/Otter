ofa_large = {
    "d_model": 1024,
    "encoder_ffn_dim": 4 * 1024,
    "encoder_layers": 12,
    "encoder_attention_heads": 16,
    "decoder_ffn_dim": 4 * 1024,
    "decoder_layers": 12,
    "decoder_attention_heads": 16,
    "resnet_type": "resnet152"
}

ofa_base = {
    "d_model": 768,
    "encoder_ffn_dim": 4 * 768,
    "encoder_layers": 6,
    "encoder_attention_heads": 12,
    "decoder_ffn_dim": 4 * 768,
    "decoder_layers": 6,
    "decoder_attention_heads": 12,
    "resnet_type": "resnet101"
}


ofa_tiny = {
    "d_model": 256,
    "encoder_ffn_dim": 4 * 256,
    "encoder_layers": 4,
    "encoder_attention_heads": 4,
    "decoder_ffn_dim": 4 * 256,
    "decoder_layers": 4,
    "decoder_attention_heads": 4,
    "resnet_type": "resnet50"
}



architecture_configs_dict = {
    "ofa-base": {
        "config": ofa_base
    },
    "ofa-large": {
        "config": ofa_large
    },
    "ofa-tiny": {
        "config": ofa_tiny
    }
}