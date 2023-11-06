import os
import random
import subprocess
import sys
from contextlib import suppress

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

try:
    from transformers.models.idefics.processing_idefics import image_attention_mask_for_packed_input_ids, incremental_to_binary_attention_mask
except ImportError:
    print("Failed to import Idefics processing module.")


def truncate_text(path, keep_start=10, keep_end=10, truncate_to="..."):
    if len(path) <= (keep_start + keep_end + len(truncate_to)):
        return path
    return path[:keep_start] + truncate_to + path[-keep_end:]


def master_print(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


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
    if args.rank != 0:
        return

    # Run pytest with the necessary arguments.
    result = subprocess.run(["pytest", "-m", "prerun", f"--yaml-path={args.training_data_yaml}"])

    if result.returncode != 0:
        print("YAML verification failed!")
        sys.exit(1)


def get_grouped_params(model, wd):
    params_with_wd, params_without_wd = [], []

    def apply_decay(x):
        return "gated_cross_attn_layer" in x and "ff_gate" not in x and "attn_gate" not in x and "norm" not in x and "bias" not in x

    for n, p in model.named_parameters():
        # if p.requires_grad:
        if apply_decay(n):
            params_with_wd.append(p)
        else:
            params_without_wd.append(p)

    return [
        {"params": params_with_wd, "weight_decay": wd},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def save_checkpoint(epoch, model, args, accelerator, unwrapped_model=None, global_step=None):
    """Save a checkpoint for the model."""
    # Ensure the directory exists
    if not os.path.exists(args.external_save_dir):
        os.makedirs(args.external_save_dir)

    if unwrapped_model is None:
        unwrapped_model = accelerator.unwrap_model(model)

    # Formulate the checkpoint filename based on whether it's an epoch or global_step checkpoint
    if global_step:
        checkpoint_path = f"{args.external_save_dir}/checkpoint_steps_{global_step}.pt"
        checkpoint_dict = {
            "steps": global_step,
            "model_state_dict": get_checkpoint(unwrapped_model),
        }
    else:
        checkpoint_path = f"{args.external_save_dir}/checkpoint_{epoch}.pt"
        checkpoint_dict = {"model_state_dict": get_checkpoint(unwrapped_model)}

    # Save the checkpoint if rank is 0
    if args.rank == 0:
        print(f"Saving checkpoint to {checkpoint_path}")
        accelerator.save(checkpoint_dict, checkpoint_path)

        # Save the model's configuration
        unwrapped_model.config.save_pretrained(args.external_save_dir)

        # Remove the previous checkpoint if required
        if args.delete_previous_checkpoint:
            if global_step:
                prev_checkpoint_path = f"{args.external_save_dir}/checkpoint_step_{global_step-args.save_steps_interval}.pt"
                if os.path.exists(prev_checkpoint_path):
                    os.remove(prev_checkpoint_path)
            elif epoch > 0:
                os.remove(f"{args.external_save_dir}/checkpoint_{epoch-1}.pt")


def save_checkpoint(checkpoint_dict, save_path, is_main_process, save_function):
    """Helper function to save the checkpoint."""
    save_function(checkpoint_dict, f"{save_path}/final_weights.pt", is_main_process=is_main_process)


def save_pretrained(component, save_path, is_main_process, save_function):
    """Helper function to save pretrained components."""
    component.save_pretrained(save_path, is_main_process=is_main_process, save_function=save_function, safe_serialization=False)


def save_final_weights(model, args, accelerator, processor=None, tokenizer=None):
    """Save final weights of the model."""
    unwrapped_model = accelerator.unwrap_model(model)
    is_main_process = accelerator.is_main_process
    save_path = args.external_save_dir
    model_name = args.model_name.lower()

    unwrapped_model.config.save_pretrained(save_path)

    if args.save_hf_model:
        save_pretrained(unwrapped_model, save_path, is_main_process, accelerator.save)

        if "idefics" in model_name or "fuyu" in model_name:
            save_pretrained(processor, save_path, is_main_process, accelerator.save)

        if "llama2" in model_name:
            save_pretrained(tokenizer, save_path, is_main_process, accelerator.save)
    else:
        # Save based on the distributed type
        if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
            checkpoint_dict = accelerator.get_state_dict(model)
        else:
            checkpoint_dict = get_checkpoint(model=unwrapped_model)

        if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
            trainable_params_name = [name for name, p in unwrapped_model.named_parameters() if p.requires_grad]
            checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if k in trainable_params_name}

        save_checkpoint(checkpoint_dict, save_path, is_main_process, accelerator.save)


def get_weights_for_dataloaders(dataloaders):
    total_samples = sum(len(dataloader.dataset) for dataloader in dataloaders)
    weights = [len(dataloader.dataset) / total_samples for dataloader in dataloaders]
    return weights


def get_next_dataloader(dataloader_iterators, weights):
    chosen_dataloader_index = np.random.choice(len(dataloader_iterators), p=weights)
    return dataloader_iterators[chosen_dataloader_index]


def find_and_remove_tokens(input_tensor, labels_tensor, attention_mask_tensor, token_id, tokenizer):
    batch_size, seq_len = input_tensor.size()

    # Create lists to store the new tensors
    new_input_list = []
    new_labels_list = []
    new_attention_mask_list = []

    # Loop over each sequence in the batch
    for i in range(batch_size):
        single_input = input_tensor[i, :]
        single_label = labels_tensor[i, :]
        single_attention_mask = attention_mask_tensor[i, :]

        # Remove the token_id
        new_single_input = torch.masked_select(single_input, single_input != token_id)
        new_single_label = torch.masked_select(single_label, single_input != token_id)
        new_single_attention_mask = torch.masked_select(single_attention_mask, single_input != token_id)

        # Append the new sequence to the list
        new_input_list.append(new_single_input)
        new_labels_list.append(new_single_label)
        new_attention_mask_list.append(new_single_attention_mask)

    # Pad sequences within each batch to match the longest sequence
    new_input = torch.nn.utils.rnn.pad_sequence(new_input_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    new_labels = torch.nn.utils.rnn.pad_sequence(new_labels_list, batch_first=True, padding_value=-100)
    new_attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask_list, batch_first=True, padding_value=0)

    return new_input, new_labels, new_attention_mask


def delete_tensors_from_dict(d):
    """Recursively delete tensors from a nested dictionary."""
    keys_to_delete = []
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            keys_to_delete.append(k)
        elif isinstance(v, list):
            new_list = [item for item in v if not isinstance(item, torch.Tensor)]
            d[k] = new_list
        elif isinstance(v, dict):
            delete_tensors_from_dict(v)
    for key in keys_to_delete:
        del d[key]
