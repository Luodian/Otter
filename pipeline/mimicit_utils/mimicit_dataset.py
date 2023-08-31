# Copyright 2023 The Otter Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import base64
from io import BytesIO
import re
import contextlib
import os
import orjson
import ijson.backends.yajl2_cffi as ijson
from PIL import ImageFile
from torchvision import transforms
import random

import sys
from PIL import Image, ImageFile

import torch
import numpy as np

from torch.utils.data import Dataset


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

FLAMINGO_MEAN = [0.481, 0.458, 0.408]
FLAMINGO_STD = [0.269, 0.261, 0.276]

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def random_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(random_state)


class MimicitDataset(Dataset):
    def __init__(
        self,
        args,
        mimicit_paths="",
        images_paths="",
        train_config_paths="",
        status_list=["past", "new"],
        task_name="DC",
    ):
        self.args = args
        self.tokenizer = args.tokenizer

        # self.max_src_length = args.max_src_length
        # self.max_tgt_length = args.max_tgt_length

        self.seed = args.seed
        self.patch_image_size = args.patch_image_size
        self.max_seq_len = args.max_seq_len

        self.epoch = 0

        self.inst_format = args.inst_format
        self.resample_frames = args.resample_frames
        self.text_data_list = ["LIMA", "MBPP", "TXT_SHAREGPT", "AL", "CAL", "TEXT_ONLY"]
        self.image_data_list = ["LA", "M3IT", "PF"]
        self.video_data_list = ["DC", "FunQA", "E4D", "TVC", "VideoQA"]
        self.wrap_sys = f"<<SYS>>\nYou are a helpful vision language assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n"

        scales = [(args.patch_image_size, args.patch_image_size)]

        self.patch_resize_transform = transforms.Compose(
            [
                transforms.Resize((args.patch_image_size, args.patch_image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
            ]
        )
        assert mimicit_paths != "", f"Error: The mimicit_paths do not get!"

        self.mimicit_paths = mimicit_paths
        self.images_paths = images_paths if images_paths != "" else [""] * len(mimicit_paths)
        self.train_config_paths = train_config_paths if train_config_paths != "" else [""] * len(mimicit_paths)
        self.status_list = status_list

        assert len(self.mimicit_paths) == len(self.images_paths) == len(self.train_config_paths) == len(self.status_list), f"metas do not have same number"

        self.dataset = {}
        self.images = {}
        self.train_data_list = []
        self.train_config = []
        self.task_name = args.task_name

        for (
            cur_mimicit_path,
            cur_images_path,
            cur_train_config_path,
            cur_status,
        ) in zip(self.mimicit_paths, self.images_paths, self.train_config_paths, self.status_list):
            # Load the dataset
            assert os.path.exists(cur_mimicit_path), f"Error: The local mimicit_path {cur_mimicit_path} not exists!"
            with open(cur_mimicit_path, "rb") as f:
                if self.dataset == {}:
                    self.dataset = orjson.loads(f.read())["data"]
                else:
                    self.dataset.update(orjson.loads(f.read())["data"])

            with open(cur_images_path, "rb") as f:
                for key, value in ijson.kvitems(f, "", use_float=True):
                    self.images[key] = value

            # Load the train_config
            if cur_train_config_path != "":
                assert os.path.exists(cur_train_config_path), f"Error: The local train_config_path {cur_train_config_path} not exists!"
                with open(cur_train_config_path, "rb") as f:
                    cache_train_config = orjson.loads(f.read())
            else:
                with open(cur_mimicit_path, "rb") as f:
                    cache_train_config = orjson.loads(f.read())["data"]
                    cache_train_config = {key: [] for key in cache_train_config.keys()}

            if cur_status == "new":
                cache_train_list = list(cache_train_config.keys())
            else:
                random.seed(0)
                cache_train_list = list(cache_train_config.keys())
                random.shuffle(cache_train_list)
                cache_train_list = cache_train_list[: int(len(cache_train_list) * args.past_subset_ration)]
            if self.train_data_list == []:
                self.train_data_list = cache_train_list
                self.train_config = cache_train_config
            else:
                self.train_data_list += cache_train_list
                self.train_config.update(cache_train_config)
            del cache_train_config
            del cache_train_list

        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

    def random_init_case(self, question):
        if len(question) == 0:
            return question

        first_letter = question[0]
        if random.choice([True, False]):
            first_letter = first_letter.upper()
        else:
            first_letter = first_letter.lower()

        return first_letter + question[1:]

    def pre_question(self, question):
        question = question.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ")
        question = self.random_init_case(question)

        question = re.sub(
            r"\s{2,}",
            " ",
            question,
        )
        question = question.lstrip("\n")
        question = question.rstrip("\n")
        question = question.strip(" ")

        return question

    def pre_answer(self, answer, max_ans_words=1024):
        answer = re.sub(
            r"\s{2,}",
            " ",
            answer,
        )
        answer = answer.rstrip("\n")
        answer = answer.strip(" ")

        # truncate question
        return_answer = ""
        answers = answer.split(".")

        for _ in answers:
            if return_answer == "":
                cur_answer = _
            else:
                cur_answer = ".".join([return_answer, _])
            if len(cur_answer.split(" ")) <= max_ans_words:
                return_answer = cur_answer
            else:
                break

        if return_answer == "":
            answer_words = answer.split(" ")
            return_answer = " ".join(answer_words[:max_ans_words])
        else:
            if return_answer[-1] != "." and return_answer != answers:
                return_answer += "."

        return return_answer

    def pre_caption(self, caption, max_words):
        caption = caption.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ").replace("<person>", "person")

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > max_words:
            caption = " ".join(caption_words[:max_words])

        return caption

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def resample_frames_fn(self, image_ids, resample_frames):
        indices = np.linspace(0, len(image_ids) - 1, resample_frames, dtype=int)
        image_ids = [image_ids[i] for i in indices]
        assert len(image_ids) == resample_frames
        return image_ids

    def process_llava(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format="simple"):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        # random.shuffle(all_instruction_ids)
        if "CONV" in instruction_id:
            for idx, cur_instruction_id in enumerate(all_instruction_ids):
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                cur_instruction = self.pre_question(cur_instruction)
                cur_answer = self.pre_answer(cur_answer)
                if inst_format == "llama2":
                    if idx == 0:
                        cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                    else:
                        cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                elif inst_format == "idefics":
                    if idx == 0:
                        cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                    elif idx < len(all_instruction_ids) - 1:
                        cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                    elif idx == len(all_instruction_ids) - 1:
                        cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>"
                elif inst_format == "simple":
                    if idx == 0:
                        cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                    else:
                        cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                all_texts += cur_text

            # if inst_format == "simple":
            #     all_texts = f"<image>{all_texts}"
            cur_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
            cur_image = self.images[cur_image_id]
            cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
            patch_images = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
        else:
            for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                cur_image = self.images[cur_instruction_image_id]
                cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
                cur_patch_image = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
                if len(patch_images) == 0:
                    patch_images = cur_patch_image
                else:
                    patch_images = torch.cat((patch_images, cur_patch_image))
                cur_instruction = self.pre_question(cur_instruction)
                cur_answer = self.pre_answer(cur_answer)
                if inst_format == "llama2":
                    cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                elif inst_format == "idefics":
                    cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                elif inst_format == "simple":
                    cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                all_texts += cur_text
        return patch_images, all_texts  # incontext_text, query_text

    def process_general_videoqa(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, resample_frames=32, inst_format="simple"):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        random.shuffle(all_instruction_ids)
        for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_instruction = self.pre_question(cur_instruction)
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_answer = self.pre_answer(cur_answer)
            if inst_format == "llama2":
                if idx == 0:
                    cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
            elif inst_format == "idefics":
                if idx == 0:
                    cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction} Assistant:<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"User:{cur_instruction} Assistant:<answer>{cur_answer}<|endofchunk|>"
            elif inst_format == "simple":
                if idx == 0:
                    cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"

            all_texts += cur_text

        # <image>User: {cur_incontext_instruction} GPT:<answer> {cur_incontext_answer}<|endofchunk|>User: {instruction} GPT:<answer> {answer}<|endofchunk|>
        # <image>User: what does the image describe? GPT: XXX <|endofchunk|>User: Do you think this image is funny GPT:<answer> YYY <|endofchunk|>
        image_ids = self.resample_frames_fn(image_ids, resample_frames)
        for cur_image_id in image_ids:
            cur_image = self.images[cur_image_id]
            cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
            cur_patch_image = self.patch_resize_transform(cur_image).unsqueeze(0)
            if len(patch_images) == 0:
                patch_images = cur_patch_image
            else:
                patch_images = torch.cat((patch_images, cur_patch_image))

        patch_images = patch_images.unsqueeze(0)
        return patch_images, all_texts

    def process_spot_the_difference(self, instruction_id, instruction, answer, image_ids, in_context_example_ids):
        patch_images = torch.tensor([])
        incontext_text = ""
        # <image>User: {instruction} GPT:<answer> {answer}<|endofchunk|>
        for cur_image_id in image_ids:
            cur_image = self.images[cur_image_id]
            cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
            cur_patch_image = self.patch_resize_transform(cur_image).unsqueeze(0)
            if len(patch_images) == 0:
                patch_images = cur_patch_image
            else:
                patch_images = torch.cat((patch_images, cur_patch_image))

        patch_images = patch_images.unsqueeze(0)
        instruction = self.pre_question(instruction)
        answer = self.pre_answer(answer)
        query_text = f"<image>User: {instruction} GPT:<answer> {answer}<|endofchunk|>"
        all_texts = f"{incontext_text}{query_text}"
        return patch_images, all_texts

    def process_scene_navigation(self, instruction_id, instruction, answer, image_ids, in_context_example_ids):
        patch_images = torch.tensor([])
        incontext_text = ""
        for cur_incontext_id in in_context_example_ids:
            cur_incontext_instruction = self.dataset[cur_incontext_id]["instruction"]
            cur_incontext_instruction = self.pre_question(cur_incontext_instruction)
            cur_incontext_answer = self.dataset[cur_incontext_id]["answer"]
            cur_incontext_answer = self.pre_answer(cur_incontext_answer)
            cur_incontext_text = f"User: {cur_incontext_instruction} GPT:<answer> {cur_incontext_answer}<|endofchunk|>"
            incontext_text += cur_incontext_text

        incontext_text = f"<image>{incontext_text}"
        # <image>User: {cur_incontext_instruction} GPT:<answer> {cur_incontext_answer}<|endofchunk|>User: {instruction} GPT:<answer> {answer}<|endofchunk|>
        for cur_image_id in image_ids:
            cur_image = self.images[cur_image_id]
            cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
            cur_patch_image = self.patch_resize_transform(cur_image).unsqueeze(0)
            if len(patch_images) == 0:
                patch_images = cur_patch_image
            else:
                patch_images = torch.cat((patch_images, cur_patch_image))

        patch_images = patch_images.unsqueeze(0)
        instruction = self.pre_question(instruction)
        answer = self.pre_answer(answer)
        query_text = f"User: {instruction} GPT:<answer> {answer}<|endofchunk|>"
        all_texts = f"{incontext_text}{all_texts}"
        return patch_images, all_texts

    def process_general_imageqa(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format="simple"):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        # the in_context_example_ids in this process_func is usually previous conversations
        for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
            cur_instruction_image_id = (
                self.dataset[cur_instruction_id]["image_ids"][0]
                if isinstance(self.dataset[cur_instruction_id]["image_ids"], list)
                else self.dataset[cur_instruction_id]["image_ids"]
            )
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_image = self.images[cur_instruction_image_id]
            try:
                cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
            except:
                print(cur_instruction_id)
                exit()
            cur_patch_image = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
            if len(patch_images) == 0:
                patch_images = cur_patch_image
            else:
                patch_images = torch.cat((patch_images, cur_patch_image))
            cur_instruction = self.pre_question(cur_instruction)
            cur_answer = self.pre_answer(cur_answer)
            if inst_format == "llama2":
                if idx == 0:
                    cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
            elif inst_format == "idefics":
                if idx == 0:
                    cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                elif idx < len(all_instruction_ids) - 1:
                    cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                elif idx == len(all_instruction_ids) - 1:
                    cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>"
            elif inst_format == "simple":
                if idx == 0:
                    cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            all_texts += cur_text
        return patch_images, all_texts

    def process_general_text(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format="simple"):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_patch_image = torch.zeros(3, 224, 224).unsqueeze(0).unsqueeze(0)
            if len(patch_images) == 0:
                patch_images = cur_patch_image
            else:
                patch_images = torch.cat((patch_images, cur_patch_image))
            cur_instruction = self.pre_question(cur_instruction)
            cur_answer = self.pre_answer(cur_answer)
            if "baize" in instruction_id:
                cur_text = f"{cur_answer}"
            elif inst_format == "llama2":
                if idx == 0:
                    cur_text = f"[INST]{self.wrap_sys} {cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
            elif inst_format == "idefics":
                cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
            elif inst_format == "simple":
                cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            all_texts += cur_text
        return patch_images, all_texts

    def process_image_text_pair(self, index):
        # try:
        cur_train_id = self.train_data_list[index]
        (
            instruction_id,
            instruction,
            answer,
            image_ids,
            in_context_example_ids,
        ) = (
            cur_train_id,
            self.dataset[cur_train_id]["instruction"],
            self.dataset[cur_train_id]["answer"],
            self.dataset[cur_train_id]["image_ids"],
            self.train_config[cur_train_id],
        )
        inst_format = self.inst_format
        resample_frames = self.resample_frames
        # self.max_src_length = self.max_tgt_length = 256

        if cur_train_id.upper().startswith("LA"):
            patch_images, all_texts = self.process_llava(instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format)
        elif cur_train_id.upper().startswith("SD") or cur_train_id.startswith("CGD"):
            patch_images, all_texts = self.process_spot_the_difference(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.upper().startswith("SN"):
            patch_images, all_texts = self.process_scene_navigation(
                instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format
            )
        elif any(cur_train_id.upper().startswith(videoqa_task) for videoqa_task in self.video_data_list) or self.task_name in self.video_data_list:
            patch_images, all_texts = self.process_general_videoqa(
                instruction_id, instruction, answer, image_ids, in_context_example_ids, resample_frames=resample_frames, inst_format=inst_format
            )
        elif any(cur_train_id.upper().startswith(text_id) for text_id in self.text_data_list) or self.task_name in self.text_data_list:
            # code to execute if cur_train_id starts with an item in self.text_data_list
            patch_images, all_texts = self.process_general_text(instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format)
        elif any(cur_train_id.upper().startswith(image_id) for image_id in self.image_data_list) or self.task_name in self.image_data_list:
            patch_images, all_texts = self.process_general_imageqa(
                instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format
            )

        all_text = self.tokenizer(
            f"{all_texts}",
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,  # for current 2k mpt/llama model, setting to 2048 causes error (2042 works)
        )

        all_item = all_text["input_ids"].squeeze(0)
        all_item_mask = all_text["attention_mask"].squeeze(0)

        all_item = torch.cat([self.bos_item, all_item, self.eos_item])
        all_item_mask = torch.cat([self.bos_mask, all_item_mask, self.eos_mask])
        # src_item = torch.cat([self.bos_item, src_item])
        # src_item_mask = torch.cat([self.bos_mask, src_item_mask])

        example = {
            "id": instruction_id,
            "source": all_item,
            "text_mask": all_item_mask,
            "patch_images": patch_images,
        }

        return example

    def __str__(self):
        return f"type: {type(self)}, length: {len(self)}"

    def __len__(self):
        return len(self.train_data_list)

    def __getitem__(self, index):
        with random_seed(self.seed, self.epoch):
            pair_sample = self.process_image_text_pair(index)
            # if dataset is not supported
            if pair_sample is None:
                return self.__getitem__(index + 1)
        return pair_sample

    def collate(self, samples):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []  # containing image-text pairs
        for sample_tuple in samples:
            samples_v1.append(sample_tuple)

        res_v1 = collate_fn(
            samples_v1,
            pad_idx=self.tokenizer.pad_token_id,
            eos_idx=self.tokenizer.eos_token_id,
        )
        return res_v1


def collate_fn(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key, pad_idx, pading_size=None):
        res = collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
            pad_to_length=pading_size,
        )
        return res

    larger_size = max([s["source"].size(0) for s in samples])

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source", pad_idx=pad_idx, pading_size=larger_size)
    src_tokens_masks = merge("text_mask", pad_idx=0, pading_size=larger_size)

    batch = {
        "id": id,
        "nsentences": len(samples),
        "net_input": {
            "input_ids": src_tokens,
            "attention_masks": src_tokens_masks,
        },
    }
    larger_incontext_num = max([s["patch_images"].size(0) for s in samples])
    if samples[0].get("patch_images", None) is not None:
        batch["net_input"]["patch_images"] = torch.stack([sample["patch_images"] for sample in samples], dim=0)

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
