import torch

if torch.__version__ < '1.2':
    mask_dtype = torch.uint8
else:
    mask_dtype = torch.bool

def is_apex_available():
    try:
        from apex import amp 
        _has_apex = True
    except ImportError:
        _has_apex = False
    return _has_apex