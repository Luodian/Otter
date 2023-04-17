# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import string
from torchvision import transforms
import base64
from io import BytesIO
from PIL import Image, ImageFile
from .transforms import *
import contextlib
from .ofa_dataset import OFADataset

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





class CaptionDataset(OFADataset):
    def __init__(self,
                 args,
                 dataset,
                 is_test=False):
        super().__init__(args, dataset, is_test)

        if args.imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((args.patch_image_size, args.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.scst = args.scst
        self.transtab = str.maketrans({key: None for key in string.punctuation})
        self.max_tgt_length = args.max_tgt_length
        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.dataset = dataset

    def __getitem__(self, index):
        uniq_id, image, caption = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        if not self.is_test and not self.scst:
            caption = caption.translate(self.transtab).strip()
            caption_token_list = caption.strip().split()
            tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
        else:
            caption = ' '.join(caption.strip().split())
            caption_list = [cap.translate(self.transtab).strip() for cap in caption.strip().split('&&')]
            tgt_caption = '&&'.join(caption_list)
        src_item = self.tokenizer(" what does the image describe?", return_tensors="pt",
                                  add_special_tokens=False).input_ids.squeeze(0)
        tgt_item = self.tokenizer(" {}".format(tgt_caption), return_tensors="pt",
                                  add_special_tokens=False).input_ids.squeeze(0)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item
        }
        return example


def get_whole_word_mask(bpe, dictionary):
    if bpe is not None:
        def is_beginning_of_word(i):
            # if i < dictionary.nspecial:
            if i < 4:
                # special elements are always considered beginnings
                return True
            tok = dictionary[i]
            # print(i, tok)
            if tok.startswith("madeupword"):
                return True
            try:
                # print(i, tok, bpe.convert_tokens_to_string(tok), bpe.convert_tokens_to_string(tok).startswith(" "))
                return bpe.convert_tokens_to_string(tok).startswith(" ")
            except ValueError:
                # print("wrong")
                return True



