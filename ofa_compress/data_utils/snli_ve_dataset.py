# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.


from torchvision import transforms
import base64
from io import BytesIO
from PIL import Image, ImageFile
from .transforms import *
import re
import logging
import warnings
from .ofa_dataset import OFADataset
from collections import defaultdict


ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class TreeNode():
    def __init__(self):
        self.child = defaultdict(TreeNode)

class Trie:

    def __init__(self, eos):
        self.root = TreeNode()
        self.eos = eos

    def insert(self, word):
        cur = self.root
        for c in word:
            cur = cur.child[c]

    def get_next_layer(self, word):
        cur = self.root
        for c in word:
            cur = cur.child.get(c)
            if cur is None:
                return [self.eos]
        return list(cur.child.keys())

class SnliVeDataset(OFADataset):
    def __init__(
            self,
            args,
            dataset,
            is_test=False):
        super().__init__(args, dataset, is_test)
        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length
        self.patch_image_size = args.patch_image_size
        self.max_image_size = args.max_image_size
        self.num_bins = args.num_bins

        self.add_caption = args.add_caption
        self.constraint_trie = args.constraint_trie
        self.prompt_type = args.prompt_type

        if args.imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        # for positioning
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((args.patch_image_size, args.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        logging.info(f"imagenet_default_mean_and_std {args.imagenet_default_mean_and_std}, mean {mean}, std {std}")
        logging.info(f"patch_image_size {self.patch_image_size}, max_image_size {self.max_image_size}")
        self.dataset = dataset

        self.src_dict = {value: key for key, value in args.tokenizer.get_vocab().items()}
        added = {value: key for key, value in args.tokenizer.get_added_vocab().items()}
        self.src_dict.update(added)

        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])

    def pre_caption(self, caption, max_words):
        caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption


    def __getitem__(self, index):
        uniq_id, image, hypothesis, caption, label = self.dataset[index]
        if label == 'contradiction':
            label = 'no'
        elif label == 'entailment':
            label = 'yes'
        elif label == 'neutral':
            label = 'maybe'
        else:
            raise NotImplementedError

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        hypothesis = self.pre_caption(hypothesis, self.max_src_length)
        src_item = self.tokenizer(' does the image describe " {} "?'.format(hypothesis), return_tensors="pt",
                                    add_special_tokens=False).input_ids.squeeze(0)
        tgt_item = self.tokenizer(" {}".format(label), return_tensors="pt",
                                    add_special_tokens=False).input_ids.squeeze(0)
        ref_dict = {label: 1.0}

        if self.add_caption:
            caption = self.pre_caption(caption, self.max_src_length)
            src_item = self.tokenizer(' can image and text1 " {} " imply text2 " {} "?'.format(caption, hypothesis), return_tensors="pt",
                                    add_special_tokens=False).input_ids.squeeze(0)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        if self.prompt_type == 'none':
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'src':
            prev_output_item = torch.cat([src_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item
        elif self.prompt_type == 'prev_output':
            prev_output_item = torch.cat([src_item[:-1], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item[:-1]
        else:
            raise NotImplementedError
        target_item[:-len(tgt_item)-1] = self.tokenizer.pad_token_id

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "decoder_prompt": decoder_prompt,
            "ref_dict": ref_dict,
        }
        if self.constraint_trie is not None:
            constraint_mask = torch.zeros((len(target_item), len(self.src_dict))).bool()
            start_idx = len(target_item) - len(tgt_item) - 1
            for i in range(len(target_item)-len(tgt_item)-1, len(target_item)):
                constraint_prefix_token = [self.tokenizer.bos_token_id] + target_item[start_idx:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            example["constraint_mask"] = constraint_mask
        return example
