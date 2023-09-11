import base64
import io
import random

import pandas as pd
from mmengine.dataset import Compose
from PIL import Image
from torch.utils.data import Dataset


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


class MMBenchDataset(Dataset):
    def __init__(self, data_file, sys_prompt="There are several options:"):
        self.df = pd.read_csv(data_file, sep="\t")
        # self.pipeline = Compose(pipeline)
        self.sys_prompt = sys_prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]["index"]
        image = self.df.iloc[idx]["image"]
        image = decode_base64_to_image(image)
        question = self.df.iloc[idx]["question"]
        answer = self.df.iloc[idx]["answer"] if "answer" in self.df.iloc[0].keys() else None
        catetory = self.df.iloc[idx]["category"]
        l2_catetory = self.df.iloc[idx]["l2-category"]

        option_candidate = ["A", "B", "C", "D", "E"]
        options = {cand: self.load_from_df(idx, cand) for cand in option_candidate if self.load_from_df(idx, cand) is not None}
        options_prompt = f"{self.sys_prompt}\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"

        hint = self.load_from_df(idx, "hint")
        data = {
            "img": image,
            "question": question,
            "answer": answer,
            "options": options_prompt,
            "category": catetory,
            "l2-category": l2_catetory,
            "options_dict": options,
            "index": index,
            "context": hint,
        }
        # data = self.pipeline(data)
        return data

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None
