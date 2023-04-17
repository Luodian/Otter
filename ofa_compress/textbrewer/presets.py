import collections

from .losses import *
from .schedulers import *
from .utils import cycle
from .projections import linear_projection, projection_with_activation

class DynamicKeyDict:
    def __init__(self, kv_dict):
        self.store = kv_dict
    def __getitem__(self, key):
        if not isinstance(key,(list,tuple)):
            return self.store[key]
        else:
            name = key[0]
            args = key[1:]
            if len(args)==1 and isinstance(args[0],dict):
                return self.store[name](**(args[0]))
            else:
                return self.store[name](*args)
    def __setitem__(self, key, value):
        self.store[key] = value
    def __contains__(self, key):
        if isinstance(key, (list,tuple)):
            return key[0] in self.store
        else:
            return key in self.store

TEMPERATURE_SCHEDULER=DynamicKeyDict(
    {'constant': constant_temperature_scheduler,
     'flsw': flsw_temperature_scheduler_builder,
     'cwsm':cwsm_temperature_scheduler_builder})
"""
(*custom dict*) used to dynamically adjust distillation temperature.

    * '**constant**' : Constant temperature.
    * '**flsw**' :  See `Preparing Lessons: Improve Knowledge Distillation with Better Supervision <https://arxiv.org/abs/1911.07471>`_. Needs parameters ``beta`` and ``gamma``.
    * '**cwsm**': See `Preparing Lessons: Improve Knowledge Distillation with Better Supervision <https://arxiv.org/abs/1911.07471>`_. Needs parameter ``beta``.

Different from other options, when using ``'flsw'`` and ``'cwsm'``, you need to provide extra parameters, for example::

    #flsw
    distill_config = DistillationConfig(
        temperature_scheduler = ['flsw', 1， 2]  # beta=1, gamma=2
    )
    
    #cwsm
    distill_config = DistillationConfig(
        temperature_scheduler = ['cwsm', 1] # beta = 1
    )

"""



FEATURES = ['hidden','attention']


ADAPTOR_KEYS = ['logits','logits_mask','losses','inputs_mask','labels'] + FEATURES
"""
(*list*) valid keys of the dict returned by the adaptor, includes:

    * '**logits**'
    * '**logits_mask**'
    * '**losses**'
    * '**inputs_mask**'
    * '**labels**'
    * '**hidden**'
    * '**attention**'
"""


KD_LOSS_MAP = {'mse': kd_mse_loss,
                'ce': kd_ce_loss}
"""
(*dict*) available KD losses

  * '**mse**' : mean squared error 
  * '**ce**': cross-entropy loss
"""

MATCH_LOSS_MAP = {'attention_mse_sum': att_mse_sum_loss,
                  'attention_mse': att_mse_loss,
                  'attention_ce_mean': att_ce_mean_loss,
                  'attention_ce': att_ce_loss,
                  'hidden_mse'    : hid_mse_loss,
                  'cos'  : cos_loss,
                  'pkd'  : pkd_loss,
                  'gram' : fsp_loss,
                  'fsp'  : fsp_loss,
                  'mmd'  : mmd_loss,
                  'nst'  : mmd_loss}
"""
(*dict*) intermediate feature matching loss functions, includes:

* :func:`attention_mse_sum <textbrewer.losses.att_mse_sum_loss>`
* :func:`attention_mse <textbrewer.losses.att_mse_loss>`
* :func:`attention_ce_mean <textbrewer.losses.att_ce_mean_loss>`
* :func:`attention_ce <textbrewer.losses.att_ce_loss>`
* :func:`hidden_mse <textbrewer.losses.hid_mseloss>`
* :func:`cos <textbrewer.losses.cos_loss>`
* :func:`pkd <textbrewer.losses.pkd_loss>`
* :func:`fsp <textbrewer.losses.fsp_loss>`, :func:`gram <textbrewer.losses.fsp_loss>`
* :func:`nst <textbrewer.losses.nst_loss>`, :func:`mmd <textbrewer.losses.nst_loss>`

See :ref:`intermediate_losses` for details.
"""

PROJ_MAP = {'linear': linear_projection,
            'relu'  : projection_with_activation('ReLU'),
            'tanh'  : projection_with_activation('Tanh')
            }
"""
(*dict*) layers used to match the different dimensions of intermediate features

  * '**linear**' : linear layer, no activation
  * '**relu**' : ReLU activation
  * '**tanh**': Tanh activation
"""

WEIGHT_SCHEDULER = {'linear_decay': linear_decay_weight_scheduler,
                    'linear_growth' : linear_growth_weight_scheduler}
"""
(dict) Scheduler used to dynamically adjust KD loss weight and hard_label_loss weight.

  * ‘**linear_decay**' : decay from 1 to 0 during the whole training process.
  * '**linear_growth**' : grow from 0 to 1 during the whole training process.
"""

#TEMPERATURE_SCHEDULER = {'constant': constant_temperature_scheduler,
#                         'flsw_scheduler': flsw_temperature_scheduler_builder(1,1)}


MAPS = {'kd_loss': KD_LOSS_MAP,
        'match_Loss': MATCH_LOSS_MAP,
        'projection': PROJ_MAP,
        'weight_scheduler': WEIGHT_SCHEDULER,
        'temperature_scheduler': TEMPERATURE_SCHEDULER}


def register_new(map_name, name, func):
    assert map_name in MAPS
    assert callable(func), "Functions to be registered is not callable"
    MAPS[map_name][name] = func


'''
Add new loss:
def my_L1_loss(feature_S, feature_T, mask=None):
    return (feature_S-feature_T).abs().mean()

MATCH_LOSS_MAP['my_L1_loss'] = my_L1_loss
'''
