import torch
from .utils import initializer_builder
from typing import List

act_dict = {}
for k,v in  torch.nn.modules.activation.__dict__.items():
    if not k.startswith('__'):
        act_dict[k] = v

def linear_projection(dim_in, dim_out):
    model = torch.nn.Linear(in_features=dim_in, out_features=dim_out, bias=True)
    initializer = initializer_builder(0.02)
    model.apply(initializer)
    return model

def projection_with_activation(act_fn):
    if type(act_fn) is str:
        assert act_fn in act_dict, f"invalid activations, please choice from {list(act_dict.keys())}"
        act_fn = act_dict[act_fn]()
    else:
        assert isinstance(act_fn,torch.nn.Module), "act_fn must be a string or module"
        act_fn = act_fn()
    def projection(dim_in, dim_out):
        model = torch.nn.Sequential(
            torch.nn.Linear(in_features=dim_in, out_features=dim_out, bias=True),
            act_fn)
        initializer = initializer_builder(0.02)
        model.apply(initializer)
        return model
    return projection