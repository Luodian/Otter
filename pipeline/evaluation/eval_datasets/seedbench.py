import os
import json
from PIL import Image
import numpy as np
import torch
from otter_ai import OtterForConditionalGeneration
import transformers
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = 100_000_000


def get_vision_x(input_data, model):
    if isinstance(input_data, Image.Image):
        if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
            vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(model.parameters()).dtype)
        else:
            image_processor = transformers.CLIPImageProcessor()
            vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image.")
    return vision_x


class SEEDBenchDataset(object):
    def load_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data

    def filter_image_only(self, data):
        filtered_data = []
        for d in data:
            if d["data_type"] == "image":
                filtered_data.append(d)
        return filtered_data

    def __init__(self, data_file, image_folder):
        json_data = self.load_json(data_file)
        self.data = self.filter_image_only(json_data["questions"])
        self.question_type = json_data["question_type"]
        self.question_type = {v: k for k, v in self.question_type.items()}
        self.image_folder = image_folder

    def get_image(self, image_id):
        path = os.path.join(self.image_folder, image_id + ".jpg")
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        data = self.data[idx]
        question = data["question"]
        image_id = data["data_id"]
        image = self.get_image(image_id)
        answer = data["answer"]
        options = [data["choice_a"], data["choice_b"], data["choice_c"], data["choice_d"]]

        data_dict = {
            "image": image,
            "question": question,
            "answer": answer,
            "options": options,
        }
        return data_dict

    def evaluate(self, model, tokenizer):
        # print("Evaluating...")
        num_correct = 0
        for data_dict in tqdm(self, total=len(self.data), desc="Evaluating"):
            image = data_dict["image"]
            question = data_dict["question"]
            answer = data_dict["answer"]
            options = data_dict["options"]

            # print(type(image))

            option_losses = []
            for option in options:
                query = f"<image>User:{question} GPT:<answer> {option}."
                label = query
                tokens = tokenizer(query, return_tensors="pt")
                input_ids = tokens["input_ids"]
                attention_mask = tokens["attention_mask"]
                with torch.no_grad():
                    vision_x = get_vision_x(image, model)
                    loss = model(vision_x=vision_x.to(model.device), lang_x=input_ids.to(model.device), attention_mask=attention_mask.to(model.device))
                option_losses.append(loss.items())

            prediction_idx = np.argmin(option_losses)
            prediction = ["A", "B", "C", "D"][prediction_idx]
            if prediction == answer:
                num_correct += 1

        accuracy = num_correct / len(self.data) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy


from transformers import IdeficsForVisionText2Text, AutoProcessor

if __name__ == "__main__":
    dataset = SEEDBenchDataset("/data/joshua/datasets/SEEDBench/SEED-Bench.json", "/data/joshua/datasets/SEEDBench/SEED-Bench-image")
    for data in dataset:
        print(data)
        break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # checkpoint = "/data/pufanyi/training_data/checkpoints/idefics-9b-instruct"
    checkpoint = "/data/pufanyi/training_data/checkpoints/OTTER-Image-MPT7B"
    model = OtterForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    dataset.evaluate(model, tokenizer)
