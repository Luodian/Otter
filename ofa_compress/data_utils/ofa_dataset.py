# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.


from torch.utils.data import Dataset
from PIL import Image, ImageFile

from .transforms import *
import contextlib

label_map = {'entailment': 0, 'not_entailment': 1}

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class OFADataset(Dataset):
    def __init__(self,
                 args,
                 dataset,
                 is_test=False):
        # Input parameters.
        self.args = args
        self.task_name = args.task
        self.dataset = dataset
        self.is_test = is_test
        self.tokenizer = args.tokenizer


    def __str__(self):
        return f"type: {type(self)}, length: {len(self)}"


    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data required for the task
        """
        return collate_fn(samples, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id)



def continuous_tense(word):
    if word in {'stand', 'walk', 'jump', 'sing', 'talk', 'cry'}:
        return word + 'ing'
    elif word in {'run', 'sit'}:
        return word + word[-1] + 'ing'
    elif word == 'lay':
        return 'lying'
    elif word == 'smile':
        return 'smiling'
    else:
        raise NotImplementedError

def collate_fn(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key, pad_idx, pading_size=None):
        res = collate_tokens([s[key] for s in samples], pad_idx, eos_idx=eos_idx, pad_to_length=pading_size)
        return res


    larger_size = max([s["source"].size(0) for s in samples])

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source", pad_idx=pad_idx, pading_size=larger_size)
    src_tokens_masks = merge('text_mask', pad_idx=0, pading_size=larger_size)


    batch = {
        "id": id,
        "nsentences": len(samples),
        "net_input": {
            "input_ids": src_tokens,
            "attention_masks": src_tokens_masks,
        },
    }
    if samples[0].get("patch_image", None) is not None:
        batch["net_input"]["patch_images"] = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    if samples[0].get("patch_mask", None) is not None:
        batch["net_input"]["patch_masks"] = torch.cat([sample['patch_mask'] for sample in samples])
    # image generation
    if samples[0].get("code_mask", None) is not None:
        batch["net_input"]["code_masks"] = torch.cat([sample['code_mask'] for sample in samples])
    if samples[0].get("code_image", None) is not None:
        batch["code_images"] = torch.cat([sample['code_image'] for sample in samples])
    # For classification tasks (i.e., VQA, SNLI-VE, GLUE)
    if samples[0].get("conf", None) is not None:
        batch["conf"] = torch.cat([s['conf'] for s in samples], dim=0)
    if samples[0].get("ref_dict", None) is not None:
        batch["ref_dict"] = np.array([s['ref_dict'] for s in samples])
    if samples[0].get("constraint_mask", None) is not None:
        batch["constraint_masks"] = merge("constraint_mask")
    if samples[0].get("decoder_prompt", None) is not None:
        batch["decoder_prompts"] = np.array([s['decoder_prompt'].tolist() for s in samples])
    # For detection and visual grounding
    if samples[0].get("w_resize_ratio", None) is not None:
        batch["w_resize_ratios"] = torch.stack([s["w_resize_ratio"] for s in samples], dim=0)
    if samples[0].get("h_resize_ratio", None) is not None:
        batch["h_resize_ratios"] = torch.stack([s["h_resize_ratio"] for s in samples], dim=0)
    if samples[0].get("region_coord", None) is not None:
        batch["region_coords"] = torch.stack([s['region_coord'] for s in samples], dim=0)

    return batch

def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if values[0].dim() == 1:
        res = values[0].new(len(values), size).fill_(pad_idx)
    elif values[0].dim() == 2:
        assert move_eos_to_beginning is False
        res = values[0].new(len(values), size, values[0].size(1)).fill_(pad_idx)
    else:
        raise NotImplementedError

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res




def get_whole_word_mask(bpe, dictionary):
    if bpe is not None:
        def is_beginning_of_word(i):
            # if i < dictionary.nspecial:
            if i < 4:
                # special elements are always considered beginnings
                return True
            tok = dictionary[i]
            if tok.startswith("madeupword"):
                return True
            try:
                return bpe.convert_tokens_to_string(tok).startswith(" ")
            except ValueError:
                return True




