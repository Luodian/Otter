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

from PIL import ImageFile
from torchvision import transforms
import random

import sys
from PIL import Image, ImageFile

import torch
import numpy as np

# from .transforms import *

# from transforms import *

# from transforms import *

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
        is_test=False,
        status_list=["past", "new"],
        # supported_data_types=["caption", "qa"],
    ):
        # super().__init__(args, is_test)

        self.args = args
        self.task_name = args.task
        self.is_test = is_test
        self.tokenizer = args.tokenizer

        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length

        self.seed = args.seed
        self.patch_image_size = args.patch_image_size

        self.epoch = 0

        self.inst_format = args.inst_format

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

        for cur_mimicit_path, cur_images_path, cur_train_config_path, cur_status in zip(
            self.mimicit_paths, self.images_paths, self.train_config_paths, self.status_list
        ):
            # Load the dataset
            assert os.path.exists(cur_mimicit_path), f"Error: The local mimicit_path {cur_mimicit_path} not exists!"
            with open(cur_mimicit_path, "rb") as f:
                if self.dataset == {}:
                    self.dataset = orjson.loads(f.read())["data"]
                else:
                    self.dataset.update(orjson.loads(f.read())["data"])

            # Load the images
            if cur_images_path != "":
                assert os.path.exists(cur_images_path), f"Error: The local images_path {cur_images_path} not exists!"
                with open(cur_images_path, "rb") as f:
                    if self.images == {}:
                        self.images = orjson.loads(f.read())
                    else:
                        self.images.update(orjson.loads(f.read()))

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

    def pre_question(self, question, max_ques_words):
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

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > max_ques_words:
            question = " ".join(question_words[:max_ques_words])

        return question

    def pre_answer(self, answer, max_ans_words):
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

    def resample_frames(self, image_ids, resample_frames):
        indices = np.linspace(0, len(image_ids) - 1, resample_frames, dtype=int)
        image_ids = [image_ids[i] for i in indices]
        assert len(image_ids) == resample_frames
        return image_ids

    def process_llavar(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format="simple"):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        wrap_sys = f"<<SYS>>\nYou are a helpful vision language assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n"
        # random.shuffle(all_instruction_ids)
        for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
            cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
            cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
            if inst_format == "llama2":
                if idx == 0:
                    # insert image to the first sentence of a conversation
                    cur_text = f"[INST]{wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
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

        # if inst_format == "simple":
        #     all_texts = f"<image>{all_texts}"
        cur_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
        cur_image = self.images[cur_image_id]
        cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
        patch_images = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
        return patch_images, all_texts  # incontext_text, query_text

    def process_llava(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format="simple"):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        wrap_sys = f"<<SYS>>\nYou are a helpful vision language assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n"
        # random.shuffle(all_instruction_ids)
        if "CONV" in instruction_id:
            for idx, cur_instruction_id in enumerate(all_instruction_ids[:]):
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
                cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
                if inst_format == "llama2":
                    if idx == 0:
                        cur_text = f"[INST]{wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                    else:
                        cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                elif inst_format == "idefics":
                    if idx == 0:
                        cur_text = (
                            f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction} Assistant:<answer>{cur_answer}<|endofchunk|>"
                        )
                    else:
                        cur_text = f"User:{cur_instruction} Assistant:<answer>{cur_answer}<|endofchunk|>"
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
                cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
                cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
                if inst_format == "llama2":
                    cur_text = f"[INST]{wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                elif inst_format == "idefics":
                    cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction} Assistant:<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                all_texts += cur_text
        # <image>User: {cur_incontext_instruction} GPT:<answer> {cur_incontext_answer}<|endofchunk|><image>User: {instruction} GPT:<answer> {answer}<|endofchunk|>
        # incontext_text = "<image>User: What does this image descibe? GPT:<answer>The children in the image, along with the rest of the family. They are Skiing. <|endofchunk|>"
        # query_text = f"<image>User: What does this image descibe? GPT:<answer>"
        # query_text = f"<image>User: {instruction} GPT:<answer>"
        # print(instruction_id, query_text, answer)
        return patch_images, all_texts  # incontext_text, query_text

    def process_dense_caption(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, resample_frames=32):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        random.shuffle(all_instruction_ids)
        for cur_instruction_id in all_instruction_ids[:]:
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
            cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            all_texts += cur_text

        all_texts = f"<image>{all_texts}"
        # <image>User: {cur_incontext_instruction} GPT:<answer> {cur_incontext_answer}<|endofchunk|>User: {instruction} GPT:<answer> {answer}<|endofchunk|>
        # <image>User: what does the image describe? GPT: XXX <|endofchunk|>User: Do you think this image is funny GPT:<answer> YYY <|endofchunk|>
        image_ids = self.resample_frames(image_ids, resample_frames)
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

    def process_tv_caption(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, resample_frames=16):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        random.shuffle(all_instruction_ids)
        for cur_instruction_id in all_instruction_ids[:]:
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
            cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            all_texts += cur_text

        all_texts = f"<image>{all_texts}"
        # <image>User: {cur_incontext_instruction} GPT:<answer> {cur_incontext_answer}<|endofchunk|>User: {instruction} GPT:<answer> {answer}<|endofchunk|>
        # <image>User: what does the image describe? GPT: XXX <|endofchunk|>User: Do you think this image is funny GPT:<answer> YYY <|endofchunk|>

        # make sure the frames are evenly sampled to certain number to enable batch processing
        image_ids = self.resample_frames(image_ids, resample_frames)
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

    def process_e4d(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, resample_frames=16):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        random.shuffle(all_instruction_ids)
        for cur_instruction_id in all_instruction_ids[:]:
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
            cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            all_texts += cur_text

        all_texts = f"<image>{all_texts}"
        # <image>User: {cur_incontext_instruction} GPT:<answer> {cur_incontext_answer}<|endofchunk|>User: {instruction} GPT:<answer> {answer}<|endofchunk|>
        # <image>User: what does the image describe? GPT: XXX <|endofchunk|>User: Do you think this image is funny GPT:<answer> YYY <|endofchunk|>
        # make sure the frames are evenly sampled to certain number to enable batch processing
        image_ids = self.resample_frames(image_ids, resample_frames)
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
        instruction = self.pre_question(instruction, self.max_src_length)
        answer = self.pre_answer(answer, self.max_tgt_length)
        query_text = f"<image>User: {instruction} GPT:<answer> {answer}<|endofchunk|>"
        all_texts = f"{incontext_text}{query_text}"
        return patch_images, all_texts

    def process_scene_navigation(self, instruction_id, instruction, answer, image_ids, in_context_example_ids):
        patch_images = torch.tensor([])
        incontext_text = ""
        for cur_incontext_id in in_context_example_ids:
            cur_incontext_instruction = self.dataset[cur_incontext_id]["instruction"]
            cur_incontext_instruction = self.pre_question(cur_incontext_instruction, self.max_src_length)
            cur_incontext_answer = self.dataset[cur_incontext_id]["answer"]
            cur_incontext_answer = self.pre_answer(cur_incontext_answer, self.max_tgt_length)
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
        instruction = self.pre_question(instruction, self.max_src_length)
        answer = self.pre_answer(answer, self.max_tgt_length)
        query_text = f"User: {instruction} GPT:<answer> {answer}<|endofchunk|>"
        all_texts = f"{incontext_text}{all_texts}"
        return patch_images, all_texts

    def process_funqa(self, instruction_id, instruction, answer, image_ids, in_context_example_ids):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        random.shuffle(all_instruction_ids)
        for cur_instruction_id in all_instruction_ids[:]:
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
            cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            all_texts += cur_text

        all_texts = f"<image>{all_texts}"
        # <image>User: {cur_incontext_instruction} GPT:<answer> {cur_incontext_answer}<|endofchunk|>User: {instruction} GPT:<answer> {answer}<|endofchunk|>
        # <image>User: what does the image describe? GPT: XXX <|endofchunk|>User: Do you think this image is funny GPT:<answer> YYY <|endofchunk|>
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

    def process_general_vqa(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format="simple"):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        wrap_sys = f"<<SYS>>\nYou are a helpful vision language assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n"
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
            cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
            cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
            if inst_format == "llama2":
                if idx == 0:
                    cur_text = f"[INST]{wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
            elif inst_format == "idefics":
                if idx == 0:
                    cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction} Assistant:<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"User:{cur_instruction} Assistant:<answer>{cur_answer}<|endofchunk|>"
            else:
                if idx == 0:
                    cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            all_texts += cur_text
        return patch_images, all_texts

    def process_text_instruction(self, instruction_id, instruction, answer, image_ids, in_context_example_ids):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        for cur_instruction_id in all_instruction_ids[:]:
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_patch_image = torch.zeros(3, 224, 224).unsqueeze(0).unsqueeze(0)
            if len(patch_images) == 0:
                patch_images = cur_patch_image
            else:
                patch_images = torch.cat((patch_images, cur_patch_image))
            cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
            cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
            if "baize" in instruction_id:
                cur_text = f"{cur_answer}"
            else:
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

        # self.max_src_length = self.max_tgt_length = 256

        if cur_train_id.startswith("LA"):
            patch_images, all_texts = self.process_llava(instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format)
        elif cur_train_id.startswith("DC"):
            patch_images, all_texts = self.process_dense_caption(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("TVC"):
            patch_images, all_texts = self.process_tv_caption(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("E4D"):
            patch_images, all_texts = self.process_e4d(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("SD") or cur_train_id.startswith("CGD"):
            patch_images, all_texts = self.process_spot_the_difference(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("SN"):
            patch_images, all_texts = self.process_scene_navigation(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("FunQA"):
            patch_images, all_texts = self.process_funqa(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("LLAVAR"):
            patch_images, all_texts = self.process_llavar(instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format)
        elif cur_train_id.startswith("TXT"):
            patch_images, all_texts = self.process_text_instruction(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        else:
            patch_images, all_texts = self.process_general_vqa(instruction_id, instruction, answer, image_ids, in_context_example_ids, inst_format=inst_format)

        src_text = self.tokenizer(
            f"{all_texts}",
            return_tensors="pt",
            add_special_tokens=False,
        )

        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])
        # src_item = torch.cat([self.bos_item, src_item])
        # src_item_mask = torch.cat([self.bos_mask, src_item_mask])

        example = {
            "id": instruction_id,
            "source": src_item,
            "text_mask": src_item_mask,
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
