# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.


import base64
from io import BytesIO
import re
import contextlib

from PIL import ImageFile
from torchvision import transforms

from .transforms import *
from .input_dataset import FileDataset


from .multi_instruct_dataset import (
    MultiInstructDataset,
    collate_fn,
)

label_map = {"entailment": 0, "not_entailment": 1}

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


class UnifyDataset(MultiInstructDataset):
    def __init__(self, args, is_test=False, supported_data_types=["caption", "qa"]):
        super().__init__(args, is_test)
        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length

        self.seed = args.pretrain_seed
        self.code_dict_size = args.code_dict_size
        self.patch_image_size = args.patch_image_size
        self.code_image_size = args.code_image_size
        self.supported_data_types = supported_data_types

        self.epoch = 0

        scales = [(args.patch_image_size, args.patch_image_size)]

        # TODO: check if random augment is correct, especially for some questions related to colors.
        self.patch_resize_transform = transforms.Compose(
            [
                RandomResize(scales),
                transforms.CenterCrop(args.patch_image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
            ]
        )

        self.file_path = args.multi_instruct_path
        assert os.path.exists(
            self.file_path
        ), "Error: The local datafile {} not exists!".format(self.file_path)
        self.separator = "\t"
        # self.selected_col_ids = [
        #         int(col_id) for col_id in args.selected_col_ids.split(",")
        #     ]
        # self.dtypes = [str for col_id in self.selected_col_ids]

        with open(self.file_path) as f:
            self.dataset = f.readlines()

        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

        self.rank = args.rank

    def pre_question(self, question, max_ques_words):
        question = (
            question.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ")
        )

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
            return_answer = " ".join(answer_words[:max_ques_words])
        else:
            if return_answer[-1] != "." and return_answer != answers:
                return_answer += "."

        return return_answer

    def pre_caption(self, caption, max_words):
        caption = (
            caption.lower()
            .lstrip(",.!?*#:;~")
            .replace("-", " ")
            .replace("/", " ")
            .replace("<person>", "person")
        )

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

    def process_image_text_pair(self, index):
        (
            uniq_id,
            image,
            caption,
            question,
            refs,
            gt_objects,
            dataset_name,
            type,
        ) = (
            self.dataset[index].rstrip("\n").split(self.separator)
        )
        if type not in self.supported_data_types:
            return None

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
        patch_image = (
            self.patch_resize_transform(image) if type != "positioning" else None
        )
        patch_mask = torch.tensor([True])
        conf = torch.tensor([1.0])
        pos_src_item = None
        neg_src_item = None

        if type == "caption":
            tgt_caption = self.pre_caption(caption, self.max_tgt_length)
            pos_src_caption = self.pre_caption(caption, self.max_src_length)

            src_text = self.tokenizer(
                " what does the image describe?",
                return_tensors="pt",
                add_special_tokens=False,
            )
            src_item = src_text["input_ids"].squeeze(0)
            src_item_mask = src_text["attention_mask"].squeeze(0)
            tgt_item = self.tokenizer(
                " {}".format(tgt_caption), return_tensors="pt", add_special_tokens=False
            ).input_ids.squeeze(0)

            pos_src_item = self.tokenizer(
                ' does the image describe " {} "?'.format(pos_src_caption),
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.squeeze(0)
            # neg_src_item = self.tokenizer(' does the image describe " {} "?'.format(neg_src_caption), return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)

        elif type == "qa":
            if dataset_name == "vqav2":
                question = self.pre_question(question, self.max_src_length)
                ref_dict = {
                    item.split("|!+")[1]: float(item.split("|!+")[0])
                    for item in refs.split("&&")
                }
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
                answer = refs.strip().replace("#", " ")
                answer = self.pre_answer(answer, self.max_tgt_length)
                conf = torch.tensor([1.0])
            elif dataset_name == "detail_23k":
                self.max_src_length = self.max_tgt_length = 256
                question = self.pre_question(question, self.max_src_length)
                question = question.strip("<image>")
                answer = refs.strip().replace("#", " ")
                answer = self.pre_answer(answer, self.max_tgt_length)
                conf = torch.tensor([1.0])
            elif dataset_name == "conversation_58k":
                self.max_src_length = self.max_tgt_length = 256
                question = self.pre_question(question, self.max_src_length)
                # caption = caption.replace("<#>"," ")
                # question = caption+" "+question.strip("<image>")
                question = question.strip("<image>")
                answer = refs.strip().replace("#", " ")
                answer = self.pre_answer(answer, self.max_tgt_length)
                conf = torch.tensor([1.0])
            # import pdb;pdb.set_trace()
            # src_text = self.tokenizer(" {}".format(question), return_tensors="pt", add_special_tokens=False)
            # src_text = self.tokenizer(f"<image>Question:{question} Answer:<answer>{answer}<|endofchunk|>", return_tensors="pt", add_special_tokens=False)
            src_text = self.tokenizer(
                f"<image>User: {question} GPT:<answer> {answer}<|endofchunk|>",
                return_tensors="pt",
                add_special_tokens=False,
            )
            src_item = src_text["input_ids"].squeeze(0)
            src_item_mask = src_text["attention_mask"].squeeze(0)
            conf = torch.tensor([conf])

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

    def __getitem__(self, index):
        with numpy_seed(self.seed, self.epoch):
            pair_samples = self.process_image_text_pair(index)
            # if dataset is not supported
            if pair_samples is None:
                return self.__getitem__(index + 1)
        return pair_samples

    def collate(self, samples):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []  # containing image-text pairs
        for sample_tuple in samples:
            samples_v1.append(sample_tuple[0])

        res_v1 = collate_fn(
            samples_v1,
            pad_idx=self.tokenizer.pad_token_id,
            eos_idx=self.tokenizer.eos_token_id,
        )
        return res_v1


if __name__ == "__main__":
    from PIL import Image, ImageFile
    from io import BytesIO
    import base64
    from tqdm import tqdm
    import json

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    test_dataset = FileDataset(
        "/home/v-boli7/projects/PET-VLM/example_unified_data/vision_language_examples.tsv",
        "0,1,2,3,4,5,6,7",
    )

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
