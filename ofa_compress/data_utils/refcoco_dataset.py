# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.


import base64
from io import BytesIO
from PIL import Image, ImageFile
from .transforms import *
import re
import logging
import warnings
from .ofa_dataset import OFADataset



ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class RefcocoDataset(OFADataset):
    def __init__(self,
                 args,
                 dataset,
                 is_test=False):
        super().__init__(args, dataset, is_test)
        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length
        self.patch_image_size = args.patch_image_size
        self.max_image_size = args.max_image_size
        self.num_bins = args.num_bins

        if args.imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        # for positioning
        self.positioning_transform = Compose([
            RandomResize([self.patch_image_size], max_size=self.patch_image_size),
            ToTensor(),
            Normalize(mean=mean, std=std, max_image_size=self.max_image_size)
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
        item = self.dataset[index]
        uniq_id, base64_str, text, region_coord = item
        image = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])


        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])
        quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
        region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
        src_caption = self.pre_caption(text, self.max_src_length)
        src_item = self.tokenizer(' which region does the text " {} " describe?'.format(src_caption), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        tgt_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(region_coord.split()))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            "region_coord": region
        }
        return example
