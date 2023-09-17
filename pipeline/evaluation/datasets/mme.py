import base64
import io
from PIL import Image
from torch.utils.data import Dataset
import json


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


class MMEDataset(Dataset):
    def load_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data

    def __init__(self, instruction_file, train_file, image_file):
        super().__init__("mme_dataset", instruction_file)
        self.instruction_file = instruction_file
        self.train_file = train_file
        self.image_file = image_file
        self.instruction_data = self.load_json(self.instruction_file)
        self.train_data = self.load_json(self.train_file)
        self.image_data = self.load_json(self.image_file)
        self.ids = list(self.instruction_data["data"].keys())

    def __len__(self):
        return len(self.instruction_data["data"])

    def __getitem__(self, idx):
        row = self.instruction_data["data"][self.ids[idx]]
        question = row["instruction"]
        answer = row["answer"]
        image_id = row["image_ids"][0]
        image = decode_base64_to_image(self.image_data[image_id])

        data = {
            "question": question,
            "answer": answer,
            "image": image,
        }
        return data
