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

class VqaGenDataset(OFADataset):
    def __init__(self,
                 args,
                 table_name,
                 tokenizer,
                 selected_cols,
                 data_slice=True,
                 capacity=512,
                 shuffle_buffer_size=8194,
                 is_test=False):
        super().__init__(args, table_name, tokenizer, selected_cols,
                         data_slice, capacity, shuffle_buffer_size, is_test)
        self.max_src_length = args.max_src_length
        self.max_object_length = args.max_object_length
        self.max_tgt_length = args.max_tgt_length
        self.patch_image_size = args.patch_image_size
        self.max_image_size = args.max_image_size
        self.num_bins = args.num_bins

        self.add_object = args.add_object
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
        if not is_test:
            table_index = args.tables.split(",").index(table_name)
            self.dataset = args.train_dataset[table_index]
        else:
            self.dataset = args.test_dataset

        self.src_dict = {value: key for key, value in tokenizer.get_vocab().items()}
        added = {value: key for key, value in tokenizer.get_added_vocab().items()}
        self.src_dict.update(added)

        self.bos_item = torch.LongTensor([tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([tokenizer.eos_token_id])

    def pre_question(self, question, max_ques_words):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        question = re.sub(
            r"\s{2,}",
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')

        # truncate question
        question_words = question.split(' ')
        if len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])

        return question


    def __getitem__(self, index):
        item = self.dataset[index]
        if len(item) == 5:
            uniq_id, image, question, ref, predict_objects = item
        else:
            uniq_id, image, question, ref, predict_objects, caption = item

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        question = self.pre_question(question, self.max_src_length)
        question = question + '?' if not question.endswith('?') else question
        src_item = self.tokenizer(" {}".format(question), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)

        ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in ref.split('&&')}
        answer = max(ref_dict, key=ref_dict.get)
        conf = torch.tensor([ref_dict[answer]])
        tgt_item = self.tokenizer(" {}".format(answer), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)


        if self.add_object and predict_objects is not None:
            predict_object_seq = ' '.join(predict_objects.strip().split('&&')[:self.max_object_length])
            predict_object_item = self.tokenizer(" object: {}".format(predict_object_seq), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
            src_item = torch.cat([src_item, predict_object_item])

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
        target_item[:-len(tgt_item) - 1] = self.tokenizer.pad_token_id

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "decoder_prompt": decoder_prompt,
            "ref_dict": ref_dict,
            "conf": conf,
        }
        if self.constraint_trie is not None:
            constraint_mask = torch.zeros((len(target_item), len(self.src_dict))).bool()
            start_idx = len(target_item) - len(tgt_item) - 1
            for i in range(len(target_item) - len(tgt_item) - 1, len(target_item)):
                constraint_prefix_token = [self.tokenizer.bos_token_id] + target_item[start_idx:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            example["constraint_mask"] = constraint_mask
        return example


