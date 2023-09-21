import os
import orjson
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def get_image(image_folder, image_id):
    image_path = os.path.join(image_folder, image_id)
    image = Image.open(image_path)
    return image


class SEEDBench(Dataset):
    def __init__(self, data_file, image_folder, sys_prompt="There are several options:"):
        super().__init__("SEEDBench", data_file)
        with open(data_file, "rb") as f:
            data = orjson.loads(f.read())["questions"]
        self.data = []
        for item in data:
            if item["data_type"] == "image" and os.path.exists(os.path.join(image_folder, item["data_id"])):
                self.data.append(item)
        if len(self.data) == 0:
            raise ValueError("No valid data found!")
        self.sys_prompt = sys_prompt
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = get_image(self.image_folder, self.data[idx]["data_id"])

        question = self.data[idx]["question"]
        answer = self.data[idx]["answer"]

        option_candidate = {
            "A": "choice_a",
            "B": "choice_b",
            "C": "choice_c",
            "D": "choice_d",
        }
        options = "\n".join([f"{key}. {self.data[idx][item]}" for key, item in option_candidate.items()])

        cur_prompt = question + "\n" + self.sys_prompt + "\n" + options

        data = {
            "question": cur_prompt,
            "answer": answer,
            "image": image,
        }
        return data

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


if __name__ == "__main__":
    dataset = SEEDBench(
        "/data/pufanyi/training_data/SEEDBench/SEED-Bench.json",
        "/data/pufanyi/training_data/SEEDBench/SEED-Bench-image",
    )
    for item in dataset:
        print(item)
        break
