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

from .transforms import *

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
        cur_mimicit_path,
        cur_images_path,
        cur_train_config_path,
        is_test=False,
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
        # self.supported_data_types = supported_data_types

        self.epoch = 0

        scales = [(args.patch_image_size, args.patch_image_size)]

        self.patch_resize_transform = transforms.Compose(
            [
                RandomResize(scales),
                transforms.CenterCrop(args.patch_image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
            ]
        )

        self.mimicit_path = cur_mimicit_path
        self.images_path = cur_images_path
        self.train_config_path = cur_train_config_path

        assert os.path.exists(cur_mimicit_path), f"Error: The local mimicit_path {cur_mimicit_path} not exists!"

        assert os.path.exists(cur_images_path), f"Error: The local images_path {cur_images_path} not exists!"

        assert os.path.exists(cur_train_config_path), f"Error: The local train_config_path {cur_train_config_path} not exists!"

        # Load the dataset
        with open(self.mimicit_path, "rb") as f:
            self.dataset = orjson.loads(f.read())["data"]

        # Load the images
        with open(self.images_path, "rb") as f:
            self.images = orjson.loads(f.read())

        # Load the train_config
        with open(self.train_config_path, "rb") as f:
            self.train_config = orjson.loads(f.read())

        self.train_data_list = list(self.train_config.keys())

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

    def process_llavar(self, instruction_id, instruction, answer, image_ids, in_context_example_ids):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        # random.shuffle(all_instruction_ids)
        for cur_instruction_id in all_instruction_ids[:]:
            cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
            cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
            cur_text = f"User: {cur_instruction} GPT:<answer> {cur_answer}<|endofchunk|>"
            all_texts += cur_text

        all_texts = f"<image>{all_texts}"
        cur_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
        cur_image = self.images[cur_image_id]
        cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
        patch_images = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
        return patch_images, all_texts  # incontext_text, query_text

    def process_llava(self, instruction_id, instruction, answer, image_ids, in_context_example_ids):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        # random.shuffle(all_instruction_ids)
        if "CONV" in instruction_id:
            for cur_instruction_id in all_instruction_ids[:]:
                cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
                cur_instruction = self.dataset[cur_instruction_id]["instruction"]
                cur_answer = self.dataset[cur_instruction_id]["answer"]
                cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
                cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
                cur_text = f"User: {cur_instruction} GPT:<answer> {cur_answer}<|endofchunk|>"
                all_texts += cur_text

            all_texts = f"<image>{all_texts}"
            cur_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
            cur_image = self.images[cur_image_id]
            cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
            patch_images = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
        else:
            for cur_instruction_id in all_instruction_ids[:]:
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
                cur_text = f"<image>User: {cur_instruction} GPT:<answer> {cur_answer}<|endofchunk|>"
                all_texts += cur_text
        # <image>User: {cur_incontext_instruction} GPT:<answer> {cur_incontext_answer}<|endofchunk|><image>User: {instruction} GPT:<answer> {answer}<|endofchunk|>
        # incontext_text = "<image>User: What does this image descibe? GPT:<answer>The children in the image, along with the rest of the family. They are Skiing. <|endofchunk|>"
        # query_text = f"<image>User: What does this image descibe? GPT:<answer>"
        # query_text = f"<image>User: {instruction} GPT:<answer>"
        # print(instruction_id, query_text, answer)
        return patch_images, all_texts  # incontext_text, query_text

    def process_dense_caption(self, instruction_id, instruction, answer, image_ids, in_context_example_ids):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        random.shuffle(all_instruction_ids)
        for cur_instruction_id in all_instruction_ids[:]:
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
            cur_text = f"User: {cur_instruction} GPT:<answer> {cur_answer}<|endofchunk|>"
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
            cur_text = f"User: {cur_instruction} GPT:<answer> {cur_answer}<|endofchunk|>"
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
            cur_text = f"User: {cur_instruction} GPT:<answer> {cur_answer}<|endofchunk|>"
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
            cur_text = f"User: {cur_instruction} GPT:<answer> {cur_answer}<|endofchunk|>"
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

    def process_general_vqa(self, instruction_id, instruction, answer, image_ids, in_context_example_ids):
        patch_images = torch.tensor([])
        all_texts = ""
        all_instruction_ids = in_context_example_ids + [instruction_id]
        for cur_instruction_id in all_instruction_ids[:]:
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
            cur_text = f"<image>User: {cur_instruction} GPT:<answer> {cur_answer}<|endofchunk|>"
            all_texts += cur_text
        return patch_images, all_texts

    def process_image_text_pair(self, index):
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

        # self.max_src_length = self.max_tgt_length = 256

        if cur_train_id.startswith("LA"):
            patch_images, all_texts = self.process_llava(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("DC"):
            patch_images, all_texts = self.process_dense_caption(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("TVC"):
            patch_images, all_texts = self.process_tv_caption(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("E4D"):
            patch_images, all_texts = self.process_e4d(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("SD"):
            patch_images, all_texts = self.process_spot_the_difference(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("SN"):
            patch_images, all_texts = self.process_scene_navigation(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("FunQA"):
            patch_images, all_texts = self.process_funqa(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        elif cur_train_id.startswith("LLAVAR"):
            patch_images, all_texts = self.process_llavar(instruction_id, instruction, answer, image_ids, in_context_example_ids)
        else:
            patch_images, all_texts = self.process_general_vqa(instruction_id, instruction, answer, image_ids, in_context_example_ids)

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
    # import pdb;pdb.set_trace()
    larger_incontext_num = max([s["patch_images"].size(0) for s in samples])
    # import pdb;pdb.set_trace()
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


if __name__ == "__main__":
    from PIL import Image, ImageFile
    from io import BytesIO
    import base64
    from tqdm import tqdm
    import json
    import argparse
    import sys

    sys.path.append("/mnt/petrelfs/zhangyuanhan/Otter/")
    from flamingo.modeling_flamingo import FlamingoForConditionalGeneration

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multi_instruct_path",
        type=str,
        help="path to multi_instruct dataset, this should be a glob pattern such as vision_language_examples.tsv",
    )
    parser.add_argument("--offline", action="store_true")

    args = parser.parse_args()

# --multi_instruct_path="/mnt/petrelfs/zhangyuanhan/data/m3it/reasoning/science_qa/scienceqa_instructions.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/gqa/gqa_instructions.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/text-vqa/text-vqa_instructions.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/vqav2/vqav2_instructions.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/a-okvqa/aokvqa_v1p0_instructions.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/LACONV_instructions.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/LACR_T2T_instructions.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/LADD_instructions.json" \
# --images_path="/mnt/petrelfs/zhangyuanhan/data/m3it/reasoning/science_qa/scienceqa_00.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/gqa/gqa_00.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/text-vqa/text-vqa_00.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/vqav2/vqav2_00.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/a-okvqa/aokvqa_v1p0_00.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/LA_00.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/LA_00.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/LA_00.json" \
# --train_config_path="/mnt/petrelfs/zhangyuanhan/data/m3it/reasoning/science_qa/scienceqa_train.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/gqa/gqa_train.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/text-vqa/text-vqa_train.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/vqav2/vqav2_train.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/a-okvqa/aokvqa_v1p0_train.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/LACONV_train.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/LACR_T2T_train.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/LADD_train.json" \

    args.multi_instruct_path = "/mnt/petrelfs/zhangyuanhan/data/mimicit/LLAVAR/LLAVAR_instructions.json"  # ,/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LACR_I2I_instructions.json,/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LACR_T2T_instructions.json,/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LADD_instructions.json"
    args.images_path = "/mnt/petrelfs/zhangyuanhan/data/mimicit/LLAVAR/LLAVAR.json"
    args.train_config_path = "/mnt/petrelfs/zhangyuanhan/data/mimicit/LLAVAR/LLAVAR_train.json"  # ,/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LACR_I2I_train.json,/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LACR_T2T_train.json,/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LADD_train.json"
    args.max_src_length = 256
    args.max_tgt_length = 256
    args.task = "pretrain"
    args.pretrain_seed = 0
    args.patch_image_size = 224

    from transformers import LlamaTokenizer

    with open("/mnt/petrelfs/zhangyuanhan/weights/flamingo_9b_hf/config.json") as f:
        config = json.load(f)

    tokenizer = LlamaTokenizer.from_pretrained("/mnt/petrelfs/zhangyuanhan/weights/llama-7b-hf")
    # add <answer> token to tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]})

    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    args.tokenizer = tokenizer

    cur_multi_instruct_path, cur_images_path, cur_train_config_path = args.multi_instruct_path, args.images_path, args.train_config_path

    test_dataset = MimicitDataset(args, cur_multi_instruct_path, cur_images_path, cur_train_config_path)

    uniq_id_dict = {}
    samples = []
    counter = 0
    for _ in tqdm(test_dataset):
        if counter > 0:
            break
        counter += 1
        samples.append(_)
    cur_data = test_dataset.collate(samples)
    import pdb

    pdb.set_trace()
    # import pdb;pdb.set_trace()
    # uniq_id, image, caption, question, refs, gt_objects, dataset_name, type = _
    # # index = random.choice(positive_caption_dict[uniq_id])
    # # prompt_uniq_id, prompt_image, prompt_caption, prompt_question, prompt_refs, prompt_gt_objects, prompt_dataset_name, prompt_type = test_dataset.get_prompt_item(int(index))
    # uniq_id, image, caption, question, refs, gt_objects, dataset_name, type = _
    # if uniq_id not in uniq_id_dict:
    #     uniq_id_dict[uniq_id] = 0

    # print(uniq_id, image, caption, question, refs, gt_objects, dataset_name, type)
