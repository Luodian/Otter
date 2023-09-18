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
import ijson.backends.yajl2_c as ijson
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


import numpy as np


def resample_data(data, N):
    # If N is equal to the length of the list, return the list
    if N == len(data):
        return data
    # Upsample if N is greater than the list length
    elif N > len(data):
        # Calculate the number of times the list has to be repeated
        repeat_times = N // len(data)
        remainder = N % len(data)

        # Create the new list by repeating the data
        upsampled_data = data * repeat_times

        # Add the remainder of the items by randomly sampling
        random.seed(0)
        upsampled_data += random.choices(data, k=remainder)

        return upsampled_data
    # Downsample if N is smaller than the list length
    else:
        random.seed(0)
        return random.sample(data, N)


class MimicitDataset(Dataset):
    def __init__(self, args, dataset_info):
        self.args = args
        self.tokenizer = args.tokenizer
        self.remove_symbols = args.remove_symbols if hasattr(args, "remove_symbols") else True
        # remove more symbols in the question and answer, make the question and answer more clean and training loss more stable.

        self.mimicit_paths = []
        self.num_samples_list = []
        self.train_config_paths = []
        self.images_paths = []
        self.task_names = []

        for key, value in dataset_info.items():
            self.task_names.append(key)
            self.mimicit_paths.append(value.get("mimicit_path", ""))
            self.num_samples_list.append(value.get("num_samples", 0))
            self.train_config_paths.append(value.get("train_config_path", ""))
            self.images_paths.append(value.get("images_path", ""))

        self.seed = args.seed
        self.patch_image_size = args.patch_image_size
        self.max_seq_len = args.max_seq_len

        self.epoch = 0

        self.instruction_format = args.instruction_format
        self.resample_frames = args.resample_frames
        self.text_data_list = [
            "LIMA",
            "MBPP",
            "SHAREGPT",
            "AL",
            "CAL",
            "TEXT_ONLY",
            "GUANACO",
            "TXT_ULTRACHAT",
            "ORCACHAT",
        ]
        self.in_context_imageqa_data_list = ["LACR_T2T", "LACR_I2I"]
        # image data list (including multi-round conv)
        self.imageqa_data_list = [
            "LACONV",
            "LADD",
            "M3IT",
            "PF",
            "PL",
            "SCIENCEQA",
            "SVIT",
            "IQA",
            "REFCOCO",
            "VQAV2",
            "OKVQA",
            "A-OKVQA",
            "GQA",
            "TEXT-VQA",
            "IMAGENET",
            "COCO",
            "COCO-GOI",
            "VSR",
        ]
        self.video_data_list = ["DC", "FunQA", "E4D", "TVC", "VideoQA", "EAI"]
        self.wrap_sys = f"<<SYS>>\nYou are a helpful vision language assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n"

        self.patch_resize_transform = transforms.Compose(
            [
                transforms.Resize(
                    (args.patch_image_size, args.patch_image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
            ]
        )
        self.status_list = status_list

        assert len(self.mimicit_paths) == len(self.images_paths) == len(self.train_config_paths) == len(self.status_list), f"metas do not have same number"

        self.dataset = {}
        self.images = {}
        self.train_data_list = []
        self.train_config = {}

        # Get the length of each dataset and use the second largest value as the length of each dataset
        # data_length_list = []
        # for cur_mimicit_path, cur_train_config_path in zip(self.mimicit_paths, self.train_config_paths):
        #     # Load the train_config
        #     if cur_train_config_path != "":
        #         assert os.path.exists(cur_train_config_path), f"Error: The local train_config_path {cur_train_config_path} not exists!"
        #         with open(cur_train_config_path, "rb") as f:
        #             cache_train_config = orjson.loads(f.read())
        #     else:
        #         with open(cur_mimicit_path, "rb") as f:
        #             cache_train_config = orjson.loads(f.read())["data"]
        #             cache_train_config = {key: [] for key in cache_train_config.keys()}

        #     cache_train_list = list(cache_train_config.keys())

        #     data_length_list.append(len(cache_train_list))

        #     del cache_train_config
        #     del cache_train_list

        # if len(data_length_list) == 1:
        #     max_items_per_dataset = max(data_length_list)
        # else:
        #     max_items_per_dataset = sorted(data_length_list, reverse=True)[1]

        for cur_mimicit_path, cur_images_path, cur_train_config_path, cur_status, sampled_examples, task_name in zip(
            self.mimicit_paths,
            self.images_paths,
            self.train_config_paths,
            self.status_list,
            self.num_samples_list,
            self.task_names,
        ):
            # Load the dataset
            assert os.path.exists(cur_mimicit_path), f"Error: The local mimicit_path {cur_mimicit_path} not exists!"
            with open(cur_mimicit_path, "rb") as f:
                cur_mimicit_data = orjson.loads(f.read())["data"]
                self.dataset.update(cur_mimicit_data)

            # Load the train_config
            if cur_train_config_path != "":
                with open(cur_train_config_path, "rb") as f:
                    cache_train_config = orjson.loads(f.read())
            else:
                cache_train_config = {key: value["rel_ins_ids"] for key, value in cur_mimicit_data.items()}

            resampled_train = resample_data(list(cache_train_config.keys()), sampled_examples)

            if args.rank == 0:
                print(f"Task: {task_name}, Status: Num_samples: {sampled_examples}")
                print(f"MIMICIT_PATH: {cur_mimicit_path}")
                print(f"TRAIN_CONFIG_PATH: {cur_train_config_path}")
                print(f"IMAGES_PATH: {cur_images_path}")

            if cur_images_path:
                with open(cur_images_path, "rb") as f:
                    images_data = orjson.loads(f.read())
                    for ins_key in resampled_train:
                        img_keys = self.dataset[ins_key]["image_ids"]
                        for img_key in img_keys:
                            self.images[img_key] = images_data[img_key]
                    # self.images.update(images_data)

            self.train_data_list.extend(resampled_train)
            self.train_config.update(cache_train_config)

        if args.rank == 0:
            print(f"Total number of trainable examples: {len(self.train_data_list)}")
            print(f"Total number of images: {len(self.images)}")
            print(f"Total number of dataset: {len(self.dataset)}")

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

    def pre_question(self, question, remove_symbols=True):
        if remove_symbols:
            # question = question.rstrip(",.!?*#:;~").lstrip(",.!?*#:;~")
            question = question.strip(" ")
            question = re.sub(r"\s{2,}", " ", question)
            question = question.lstrip("\n")
            question = question.rstrip("\n")
        question = question.strip(" ")

        return question

    def pre_answer(self, answer, remove_symbols=True):
        if remove_symbols:
            answer = answer.strip(" ")
            answer = re.sub(r"\s{2,}", " ", answer)
            answer = answer.lstrip("\n")
            answer = answer.rstrip("\n")
        answer = answer.strip(" ")

        # # truncate question
        # return_answer = ""
        # answers = answer.split(".")

        # for _ in answers:
        #     if return_answer == "":
        #         cur_answer = _
        #     else:
        #         cur_answer = ".".join([return_answer, _])
        #     if len(cur_answer.split(" ")) <= max_ans_words:
        #         return_answer = cur_answer
        #     else:
        #         break

        # if return_answer == "":
        #     answer_words = answer.split(" ")
        #     return_answer = " ".join(answer_words[:max_ans_words])
        # else:
        #     if return_answer[-1] != "." and return_answer != answers:
        #         return_answer += "."
        return answer

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def resample_frames_fn(self, image_ids, resample_frames):
        indices = np.linspace(0, len(image_ids) - 1, resample_frames, dtype=int)
        image_ids = [image_ids[i] for i in indices]
        assert len(image_ids) == resample_frames
        return image_ids

    def process_in_context_imageqa(
        self,
        instruction_id,
        instruction,
        answer,
        image_ids,
        in_context_example_ids,
        instruction_format="simple",
    ):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
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
            if instruction_format == "llama2":
                cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
            elif instruction_format == "idefics":
                cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
            elif instruction_format == "simple":
                cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            all_texts += cur_text

        all_texts = all_texts.rstrip("\n")  # remove the last \n
        return patch_images, all_texts  # incontext_text, query_text

    def process_general_videoqa(
        self,
        instruction_id,
        instruction,
        answer,
        image_ids,
        in_context_example_ids,
        resample_frames=32,
        instruction_format="simple",
    ):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        random.shuffle(all_instruction_ids)
        for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_instruction = self.pre_question(cur_instruction, remove_symbols=self.remove_symbols)
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_answer = self.pre_answer(cur_answer, remove_symbols=self.remove_symbols)
            if instruction_format == "llama2":
                if idx == 0:
                    cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
            elif instruction_format == "idefics":
                if idx == 0:
                    cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                elif idx < len(all_instruction_ids) - 1:
                    cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                elif idx == len(all_instruction_ids) - 1:
                    cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>"
            elif instruction_format == "simple":
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

    def process_general_imageqa(
        self,
        instruction_id,
        instruction,
        answer,
        image_ids,
        in_context_example_ids,
        instruction_format="simple",
    ):
        # including multi-round conv for single image
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_instruction = self.pre_question(cur_instruction)
            cur_answer = self.pre_answer(cur_answer)
            if instruction_format == "llama2":
                if idx == 0:
                    cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
            elif instruction_format == "idefics":
                if idx == 0:
                    cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
                else:
                    cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
            elif instruction_format == "simple":
                if idx == 0:
                    cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            all_texts += cur_text

        all_texts = all_texts.rstrip("\n")  # remove the last \n
        cur_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
        cur_image = self.images[cur_image_id]
        cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
        patch_images = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
        return patch_images, all_texts

    def process_general_text(
        self,
        instruction_id,
        instruction,
        answer,
        image_ids,
        in_context_example_ids,
        instruction_format="simple",
    ):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        patch_images = torch.zeros(3, 224, 224).unsqueeze(0).unsqueeze(0)
        for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_instruction = self.pre_question(cur_instruction)
            cur_answer = self.pre_answer(cur_answer)
            if instruction_format == "llama2":
                if idx == 0:
                    cur_text = f"[INST]{self.wrap_sys} {cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
            elif instruction_format == "idefics":
                cur_text = f"User:{cur_instruction}<end_of_utterance>\nAssistant:<answer>{cur_answer}<end_of_utterance>\n"
            elif instruction_format == "simple":
                cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            all_texts += cur_text

        all_texts = all_texts.rstrip("\n")  # remove the last \n
        return patch_images, all_texts

    def process_image_text_pair(self, index):
        # try:
        cur_train_id = self.train_data_list[index]
        (instruction_id, instruction, answer, image_ids, in_context_example_ids) = (
            cur_train_id,
            self.dataset[cur_train_id]["instruction"],
            self.dataset[cur_train_id]["answer"],
            self.dataset[cur_train_id]["image_ids"],
            self.train_config[cur_train_id],
        )
        instruction_format = self.instruction_format
        resample_frames = self.resample_frames

        if cur_train_id.upper().startswith("SD") or cur_train_id.startswith("CGD"):
            patch_images, all_texts = self.process_spot_the_difference(
                instruction_id,
                instruction,
                answer,
                image_ids,
                in_context_example_ids,
                instruction_format=instruction_format,
            )
        elif cur_train_id.upper().startswith("SN"):
            patch_images, all_texts = self.process_scene_navigation(
                instruction_id,
                instruction,
                answer,
                image_ids,
                in_context_example_ids,
                instruction_format=instruction_format,
            )
        elif any(cur_train_id.upper().startswith(videoqa_task) for videoqa_task in self.video_data_list):
            patch_images, all_texts = self.process_general_videoqa(
                instruction_id,
                instruction,
                answer,
                image_ids,
                in_context_example_ids,
                resample_frames=resample_frames,
                instruction_format=instruction_format,
            )
        elif any(cur_train_id.upper().startswith(text_id) for text_id in self.text_data_list):
            patch_images, all_texts = self.process_general_text(
                instruction_id,
                instruction,
                answer,
                image_ids,
                in_context_example_ids,
                instruction_format=instruction_format,
            )
        elif any(cur_train_id.upper().startswith(imageqa_task) for imageqa_task in self.imageqa_data_list):
            patch_images, all_texts = self.process_general_imageqa(
                instruction_id,
                instruction,
                answer,
                image_ids,
                in_context_example_ids,
                instruction_format=instruction_format,
            )
        elif any(cur_train_id.upper().startswith(in_context_imageqa_task) for in_context_imageqa_task in self.in_context_imageqa_data_list):
            patch_images, all_texts = self.process_in_context_imageqa(
                instruction_id,
                instruction,
                answer,
                image_ids,
                in_context_example_ids,
                instruction_format=instruction_format,
            )
        else:
            raise NotImplementedError(f"Error: The task {cur_train_id} is not supported!")

        all_text = self.tokenizer(
            all_texts,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,  # for current 2k mpt/llama model, setting to 2048 causes error (2042 works)
        )
        num_tokens = all_text["input_ids"].shape[1]
        if num_tokens == self.max_seq_len:
            if self.args.rank == 0:
                print(f"{cur_train_id}'s all_texts reaches the max_seq_len.")
                print(all_texts)

        all_item = all_text["input_ids"].squeeze(0)
        all_item_mask = all_text["attention_mask"].squeeze(0)

        all_item = torch.cat([self.bos_item, all_item, self.eos_item])
        all_item_mask = torch.cat([self.bos_mask, all_item_mask, self.eos_mask])

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

    ids = [s["id"] for s in samples]
    src_tokens = merge("source", pad_idx=pad_idx, pading_size=larger_size)
    src_tokens_masks = merge("text_mask", pad_idx=0, pading_size=larger_size)

    batch = {
        "id": ids,
        "nsentences": len(samples),
        "net_input": {
            "input_ids": src_tokens,
            "attention_masks": src_tokens_masks,
        },
    }
    # larger_incontext_num = max([s["patch_images"].size(0) for s in samples])
    try:
        if samples[0].get("patch_images", None) is not None:
            batch["net_input"]["patch_images"] = torch.stack([sample["patch_images"] for sample in samples], dim=0)
    except Exception as e:
        print(f"Error: {e}")
        print(batch["id"])
        exit()

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
