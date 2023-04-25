# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from torchvision import transforms
import base64
from io import BytesIO
from PIL import ImageFile

from .vision_helper import RandomAugment
from .transforms import *
from .input_dataset import FileDataset
import math
import re
import logging
import contextlib


from .ofa_dataset import OFADataset, get_whole_word_mask, continuous_tense, collate_fn

label_map = {'entailment': 0, 'not_entailment': 1}

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

FLAMINGO_MEAN = [0.481, 0.458, 0.408]
FLAMINGO_STD = [0.269, 0.261, 0.276]

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


class UnifyDataset(OFADataset):
    def __init__(self,
                 args,
                 dataset,
                 is_test=False,
                 supported_data_types=['caption', 'qa']):
        super().__init__(args, dataset, is_test)
        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length

        self.seed = args.pretrain_seed
        self.code_dict_size = args.code_dict_size
        # self.num_bins = args.num_bins
        self.patch_image_size = args.patch_image_size
        self.code_image_size = args.code_image_size
        self.supported_data_types = supported_data_types

        # self.pure_text_dataset = args.pure_text_dataset
        # self.pure_image_dataset = args.pure_image_dataset
        # self.detection_dataset = args.detection_dataset
        self.epoch = 0

        # self.all_object_list = args.all_object_list
        # self.all_caption_list = args.all_caption_list
        # self.all_relation_list = args.all_relation_list
        # self.type2ans_dict = args.type2ans_dict
        # self.ans2type_dict = args.ans2type_dict

        # self.attr2type_dict = args.attr2type_dict
        # self.type2attr_dict = args.type2attr_dict

        # self.rel2cap = args.rel2cap
        # self.rel2question = args.rel2question

        self.remove_grounded_captioning = args.remove_grounded_captioning = False
        self.remove_visual_grounding = args.remove_visual_grounding = False

        self.mask_ratio = args.mask_ratio
        self.random_ratio = args.random_ratio
        self.keep_ratio = args.keep_ratio
        self.mask_length = args.mask_length
        self.poisson_lambda = args.poisson_lambda
        self.replace_length = args.replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if self.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={self.mask_length}")
        if self.mask_length == "subword" and self.replace_length not in [0, 1]:
            raise ValueError(f"if using subwords, use replace-length=1 or 0")

        self.mask_idx = args.tokenizer.mask_token_id

        self.src_dict = {value: key for key, value in args.tokenizer.get_vocab().items()}
        added = {value: key for key, value in args.tokenizer.get_added_vocab().items()}
        self.src_dict.update(added)

        self.mask_whole_word = (
            get_whole_word_mask(args.tokenizer, self.src_dict)
            if self.mask_length != "subword"
            else None
        )
        self.mask_span_distribution = None
        if self.mask_length == "span-poisson":
            _lambda = self.poisson_lambda
            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        # self.pos_tgt_item = self.encode_text(" yes")
        self.pos_tgt_item = self.tokenizer(" yes", return_tensors="pt",
                                      add_special_tokens=False).input_ids.squeeze(0)

        # self.neg_tgt_item = self.encode_text(" no")
        self.neg_tgt_item = self.tokenizer(" no", return_tensors="pt",
                                           add_special_tokens=False).input_ids.squeeze(0)

        self.mask_left = self.mask_top = int(0.5 * self.code_image_size)
        self.mask_right = self.mask_bottom = int(1.5 * self.code_image_size)
        self.mask_ids = [
            i*self.code_image_size*2+j
            for i in range(self.code_image_size*2) for j in range(self.code_image_size*2)
            if not (self.mask_left <= i < self.mask_right and self.mask_top <= j < self.mask_bottom)
        ]

        # scales = np.arange(args.patch_image_size, 481).tolist()
        # scales = np.arange(args.patch_image_size, args.patch_image_size).tolist()
        scales = [(args.patch_image_size, args.patch_image_size)]

        # for image-text pair
        # self.detection_large_resolution_transform = Compose([
        #     RandomHorizontalFlip(),
        #     LargeScaleJitter(output_size=args.patch_image_size, aug_scale_min=1.0, aug_scale_max=1.5),
        #     ToTensor(),
        #     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=args.max_image_size)
        # ])

        # TODO: check if random augment is correct, especially for some questions related to colors.
        self.patch_resize_transform = transforms.Compose([
            RandomResize(scales),
            transforms.CenterCrop(args.patch_image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD)
        ])
        # # for pure image
        # self.patch_crop_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])
        # # for detection
        # self.detection_transform = Compose([
        #     RandomHorizontalFlip(),
        #     LargeScaleJitter(output_size=self.code_image_size*2, aug_scale_min=1.0, aug_scale_max=1.5),
        #     ToTensor(),
        #     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=args.max_image_size)
        # ])
        # # for visual grounding
        # self.positioning_transform = self.visual_grounding_transform = Compose([
        #     RandomResize(scales, max_size=672),
        #     ObjectCenterCrop((args.patch_image_size, args.patch_image_size)),
        #     ToTensor(),
        #     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=args.max_image_size)
        # ])
        self.dataset = dataset

        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

        self.rank =args.rank

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

    def pre_answer(self, answer, max_ans_words):
        answer = re.sub(
            r"\s{2,}",
            ' ',
            answer,
        )
        answer = answer.rstrip('\n')
        answer = answer.strip(' ')

        # truncate question
        return_answer = ""
        answers = answer.split('.')
        
        for _ in answers:
            if return_answer == "":
                cur_answer = _
            else:
                cur_answer = ".".join([return_answer, _])
            if len(cur_answer.split(' ')) <= max_ans_words:
                return_answer = cur_answer
            else:
                break

        if return_answer == "":
            answer_words = answer.split(' ')
            return_answer = ' '.join(answer_words[:max_ques_words])
        else:
            if return_answer[-1] != "." and return_answer != answers:
                return_answer += "."
            
        return return_answer

        
        for _ in answers:
            if return_answer == "":
                cur_answer = _
            else:
                cur_answer = ".".join([return_answer, _])
            if len(cur_answer.split(' ')) <= max_ans_words:
                return_answer = cur_answer
            else:
                break

        if return_answer == "":
            answer_words = answer.split(' ')
            return_answer = ' '.join(answer_words[:max_ques_words])
        else:
            if return_answer[-1] != "." and return_answer != answers:
                return_answer += "."
            
        return return_answer


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

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def get_negative_object(self, object):
        negative_object = random.choice(self.all_object_list[:-1])
        negative_object = self.all_object_list[-1] if negative_object == object else negative_object
        return negative_object

    def get_negative_attribute(self, attr_value):
        neg_attr_type = self.attr2type_dict[attr_value]
        neg_attr_list = self.type2attr_dict[neg_attr_type]
        neg_attr_value = random.choice(neg_attr_list[:-1])
        neg_attr_value = neg_attr_list[-1] if neg_attr_value == attr_value else neg_attr_value
        return neg_attr_value

    def get_negative_relation(self, gt_relation_set):
        negative_relation_list = [
           negative_relation for negative_relation in self.all_relation_list if negative_relation not in gt_relation_set
        ]
        negative_relation = random.choice(negative_relation_list)
        return negative_relation

    def get_negative_caption(self, caption, overlap_objects=None):
        prob = random.random()
        if overlap_objects is not None and prob > 0.6:
            overlap_object = random.choice(overlap_objects.strip().split('&&'))
            negative_object = random.choice(self.all_object_list[:-1])
            negative_object = self.all_object_list[-1] if negative_object == overlap_object else negative_object
            negative_caption = caption.replace(overlap_object, negative_object)
        else:
            negative_caption = random.choice(self.all_caption_list)
        return negative_caption

    def get_negative_answer(self, answer, conf):
        prob = random.random()
        if conf > (prob + 0.1) and answer in self.ans2type_dict:
            negative_answer_type = self.ans2type_dict[answer]
            if negative_answer_type == 'how many' and answer.isdigit() and prob > 0.5:
                negative_answer = int(answer) + random.choice([-1, 1]) if answer != 0 else 1
            else:
                negative_answer_list = self.type2ans_dict[negative_answer_type]
                negative_answer = random.choice(negative_answer_list[:-1])
                negative_answer = negative_answer_list[-1] if negative_answer == answer else negative_answer
            return negative_answer

        negative_answer_list = self.type2ans_dict['other']
        negative_answer = random.choice(negative_answer_list[:-1])
        negative_answer = negative_answer_list[-1] if negative_answer == answer else negative_answer
        return negative_answer


    def process_image_text_pair(self, index):
        uniq_id, image, caption, question, refs, gt_objects, dataset_name, type = self.dataset[index]
        if type not in self.supported_data_types:
            return None
        # uniq_id, caption, question, refs, gt_objects, predict_objects, \
        # overlap_objects, attribute, relation, image, dataset_name, type = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
        patch_image = self.patch_resize_transform(image) if type != 'positioning' else None
        patch_mask = torch.tensor([True])
        conf = torch.tensor([1.0])
        pos_src_item = None
        neg_src_item = None

        if type == 'caption':
            tgt_caption = self.pre_caption(caption, self.max_tgt_length)
            pos_src_caption = self.pre_caption(caption, self.max_src_length)
            # neg_src_caption = self.pre_caption(self.get_negative_caption(caption, gt_objects), self.max_src_length)

            src_text = self.tokenizer(" what does the image describe?", return_tensors="pt", add_special_tokens=False)
            src_item = src_text['input_ids'].squeeze(0)
            src_item_mask = src_text['attention_mask'].squeeze(0)
            tgt_item = self.tokenizer(" {}".format(tgt_caption), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)

            pos_src_item = self.tokenizer(' does the image describe " {} "?'.format(pos_src_caption), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
            # neg_src_item = self.tokenizer(' does the image describe " {} "?'.format(neg_src_caption), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)


        elif type == 'qa':
            if dataset_name == "vqav2":
                question = self.pre_question(question, self.max_src_length)
                ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in refs.split('&&')}
                answer = max(ref_dict, key=ref_dict.get)
                conf = ref_dict[answer]
            elif dataset_name == "gqa":
                question = self.pre_question(question, self.max_src_length)
                answer = refs.strip()
                conf = torch.tensor([1.0])
            elif dataset_name == "complex_reasoning_77k":
                self.max_src_length = self.max_tgt_length = 256
                question = self.pre_question(question, self.max_src_length)
                question = question.strip("<image>")
                answer = refs.strip().replace("#"," ")
                answer = self.pre_answer(answer,self.max_tgt_length)
                conf = torch.tensor([1.0])
            elif dataset_name == "detail_23k":
                self.max_src_length = self.max_tgt_length = 256
                question = self.pre_question(question, self.max_src_length)
                question = question.strip("<image>")
                answer = refs.strip().replace("#"," ")
                answer = self.pre_answer(answer,self.max_tgt_length)
                conf = torch.tensor([1.0])
                # caption = caption.replace("<#>"," ")
                # question = caption+" "+question.strip("<image>")
                question = question.strip("<image>")
                answer = refs.strip().replace("#"," ")
                answer = self.pre_answer(answer,self.max_tgt_length)
                conf = torch.tensor([1.0])
            # src_text = self.tokenizer(" {}".format(question), return_tensors="pt", add_special_tokens=False)
            # src_text = self.tokenizer(f"<image>Question:{question} Answer:<answer>{answer}<|endofchunk|>", return_tensors="pt", add_special_tokens=False)
            src_text = self.tokenizer(f"<image>{question}<answer>{answer}<|endofchunk|>", return_tensors="pt", add_special_tokens=False)
            src_item = src_text['input_ids'].squeeze(0)
        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])

        example = {
            "id": uniq_id,
            "source": src_item,
            "text_mask": src_item_mask,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "conf": conf,
        }

        examples = [example]

        return examples


    def process_pure_text(self, index):
        patch_image = torch.zeros((3, self.code_image_size*2, self.code_image_size*2))
        patch_mask = torch.tensor([False])
        code_mask = torch.tensor([False])
        conf = torch.tensor([2.0])

        examples = []
        for _ in range(2):
            uniq_id, text = self.pure_text_dataset[index]
            text = text.strip().lower()
            text_item = self.tokenizer(" {}".format(text), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)[:512]

            text_item = text_item[-256:]
            text_item = torch.cat([self.bos_item, text_item, self.eos_item])
            mask_text_item = self.add_whole_word_mask(text_item.clone(), self.mask_ratio)
            prefix_item = self.tokenizer(' what is the complete text of " "?', return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
            src_item = torch.cat([prefix_item[:-2], mask_text_item[1:-1], prefix_item[-2:]])
            tgt_item = text_item[1:-1]
            src_item = torch.cat([self.bos_item, src_item, self.eos_item])
            target_item = torch.cat([tgt_item, self.eos_item])
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            example = {
                "id": uniq_id,
                "source": src_item,
                "patch_image": patch_image,
                "patch_mask": patch_mask,
                "code_mask": code_mask,
                "target": target_item,
                "prev_output_tokens": prev_output_item,
                "conf": conf,
            }
            examples.append(example)

        return examples

    def process_pure_image(self, index):
        image_id, image, text, code, dataset_name = self.pure_image_dataset[index]
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
        patch_image = self.patch_crop_transform(image)
        patch_image[:, 64:192, 64:192] = 0
        patch_mask = torch.tensor([True])
        if dataset_name in ('imagenet_22k', 'yfcc100m', 'oi'):
            src_item = self.tokenizer(" what is the image in the middle part?", return_tensors="pt",
                           add_special_tokens=False).input_ids.squeeze(
                0)
        else:
            caption = self.pre_caption(text, self.max_src_length)
            src_item = self.tokenizer(" what is the image in the middle part? caption: {}".format(caption), return_tensors="pt",
                           add_special_tokens=False).input_ids.squeeze(
                0)
        image_code = torch.LongTensor([int(num) for num in code.strip().split()])
        tgt_item = image_code + len(self.src_dict) - self.code_dict_size - self.num_bins
        code_mask = torch.tensor([True])
        conf = torch.tensor([2.0])

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": image_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }
        return [example]

    def process_detection(self, index):
        image_id, image, label = self.detection_dataset[index]
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")

        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        label_list = label.strip().split('&&')
        for label in label_list:
            x0, y0, x1, y1, cat_id, cat = label.strip().split(',', 5)
            boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
            boxes_target["labels"].append(cat)
            boxes_target["area"].append((float(x1) - float(x0)) * (float(y1) - float(y0)))
        boxes_target["boxes"] = torch.tensor(boxes_target["boxes"])
        boxes_target["labels"] = np.array(boxes_target["labels"])
        boxes_target["area"] = torch.tensor(boxes_target["area"])

        patch_image, boxes_target = self.detection_transform(image, boxes_target)
        patch_mask = torch.tensor([True])
        code_mask = torch.tensor([False])
        conf = torch.tensor([2.0])

        quant_boxes = []
        for i, box in enumerate(boxes_target["boxes"]):
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
            quant_boxes.extend(self.tokenizer.tokenize(' {}'.format(boxes_target["labels"][i])))
        src_item = self.tokenizer(' what are the objects in the image?', return_tensors="pt", add_special_tokens=False).input_ids.squeeze(
            0)
        tgt_item = torch.tensor(self.tokenizer.convert_tokens_to_ids(quant_boxes))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": image_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }
        return [example]

    def __getitem__(self, index):
        with numpy_seed(self.seed, self.epoch):
            pair_samples = self.process_image_text_pair(index)
            # if dataset is not supported
            if pair_samples is None:
                return self.__getitem__(index + 1)
            # extra_samples = []
            # if not self.is_test:
            #     extra_samples += self.process_pure_text(0) if self.pure_text_dataset else []
            #     extra_samples += self.process_pure_image(0) if self.pure_image_dataset else []
            #     extra_samples += self.process_detection(0) if self.detection_dataset else []
        return pair_samples

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                # 4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                4, self.tokenizer.vocab_size, size=(mask_random.sum(),)
            )

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        # 4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                        4, self.tokenizer.vocab_size, size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        # 4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                        4, self.tokenizer.vocab_size, size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=4, high=self.tokenizer.vocab_size, size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    def collate(self, samples):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []   # containing image-text pairs
        # samples_v2 = []   # containing detection data, text data and image data
        for sample_tuple in samples:
            samples_v1.append(sample_tuple[0])
            # samples_v2 += sample_tuple[1]
        # if samples_v2 == []:
        #     samples_v2 += self.process_pure_text(0) if self.pure_text_dataset else []
        #     samples_v2 += self.process_pure_image(0) if self.pure_image_dataset else []
        #     samples_v2 += self.process_detection(0) if self.detection_dataset else []

        res_v1 = collate_fn(samples_v1, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id)
        # res_v2 = collate_fn(samples_v2, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id)
        return res_v1


if __name__ == '__main__':
    from PIL import Image, ImageFile
    from io import BytesIO
    import base64
    from tqdm import tqdm
    import json
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    test_dataset = FileDataset("/home/v-boli7/projects/PET-VLM/example_unified_data/vision_language_examples.tsv", "0,1,2,3,4,5,6,7")
    
    uniq_id_dict = {}
    for _ in tqdm(test_dataset):
        uniq_id, image, caption, question, refs, gt_objects, dataset_name, type = _
        # index = random.choice(positive_caption_dict[uniq_id])
        # prompt_uniq_id, prompt_image, prompt_caption, prompt_question, prompt_refs, prompt_gt_objects, prompt_dataset_name, prompt_type = test_dataset.get_prompt_item(int(index))
        uniq_id, image, caption, question, refs, gt_objects, dataset_name, type = _
        if uniq_id not in uniq_id_dict:
            uniq_id_dict[uniq_id] = 0

        print(uniq_id, image, caption, question, refs, gt_objects, dataset_name, type)

        # test_dataset.
        # niq_id, image, caption = _
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")]