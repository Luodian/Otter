from .ofa_losses import *
from textbrewer.presets import *



FEATURES = ['hidden', 'attention', 'kvs',
            'encoder_hidden', 'encoder_attention',
            'decoder_hidden', 'decoder_attention',
            'decoder_kvs', 'decoder_kvs',
            'cross_attention'
            ]

ADAPTOR_KEYS = ['logits', 'logits_mask', 'losses', 'inputs_mask', 'labels'
                ] + FEATURES

KD_LOSS_MAP['ce_with_mask'] = kd_ce_loss_with_mask


MAPS = {
    'kd_loss': KD_LOSS_MAP,
    'match_Loss': MATCH_LOSS_MAP,
    'projection': PROJ_MAP,
    'weight_scheduler': WEIGHT_SCHEDULER,
    'temperature_scheduler': TEMPERATURE_SCHEDULER
}


