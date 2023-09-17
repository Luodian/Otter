import subprocess
import sys
import time
from contextlib import suppress

import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    from transformers.models.idefics.processing_idefics import image_attention_mask_for_packed_input_ids, incremental_to_binary_attention_mask
except ImportError:
    print("Failed to import Idefics processing module.")


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif precision == "fp16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        return suppress


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict


def get_checkpoint_deepspeed_zero3(args, model):
    state_dict = {}

    for name, p in model.named_parameters():
        if p.requires_grad:
            state_dict[name] = p.data
    return state_dict

    # if torch.distributed.get_rank() == 0:
    #     # 有参数
    #     print(device_id, f"IDEFICS Trainable Params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e9:.3f} B")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)


# supporting idefics processing
def get_image_attention_mask(output_input_ids, max_num_images, tokenizer, include_image=True):
    # image_attention_mask, _ = image_attention_mask_for_packed_input_ids(output_input_ids, tokenizer)
    # image_attention_mask = incremental_to_binary_attention_mask(image_attention_mask, num_classes=max_num_images)
    if include_image:
        image_attention_mask, _ = image_attention_mask_for_packed_input_ids(output_input_ids, tokenizer)
        image_attention_mask = incremental_to_binary_attention_mask(image_attention_mask, num_classes=max_num_images)
    else:
        # in full language mode we set the image mask to all-0s
        image_attention_mask = torch.zeros(output_input_ids.shape[0], output_input_ids.shape[1], 1, dtype=torch.bool)
    return image_attention_mask

def verify_yaml(args):
    # Run pytest with the necessary arguments.
    result = subprocess.run([
        'pytest', 
        '-m', 
        'prerun', 
        f'--yaml-path={args.training_data_yaml}'
    ])
    
    if result.returncode != 0:
        print("YAML verification failed!")
        sys.exit(1)