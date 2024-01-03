import mimetypes
import os
from io import BytesIO
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image

from src.otter_ai import OtterForConditionalGeneration
from .base_model import BaseModel

from torchvision import transforms

# Disable warnings
requests.packages.urllib3.disable_warnings()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_pil_image(raw_image_data) -> Image.Image:
    # import pdb;pdb.set_trace()
    if isinstance(raw_image_data, Image.Image):
        return raw_image_data.convert('RGB')
    else:
        return Image.open(BytesIO(raw_image_data["bytes"])).convert('RGB')


def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def get_formatted_forward_prompt(question: str, answer: str) -> str:
    return f"<image>User: {question} GPT:<answer> {answer}"



def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def process_image(image, image_processor):
    image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    return image

class OtterImageLlava(BaseModel):
    def __init__(self, model_path="luodian/OTTER-Image-MPT7B", load_bit="bf16"):
        super().__init__("otter", model_path)
        precision = {}
        if load_bit == "bf16":
            precision["torch_dtype"] = torch.bfloat16
        elif load_bit == "fp16":
            precision["torch_dtype"] = torch.float16
        elif load_bit == "fp32":
            precision["torch_dtype"] = torch.float32
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
        self.model = OtterForConditionalGeneration.from_pretrained(model_path, device_map="auto", **precision)
        self.image_processor = self.model.image_processor
        dirname = os.path.dirname(model_path) 
        state_dict = {}
        counter = 0
        # for _ in os.listdir(model_path):
        #     if "pytorch_model" in _ and "-of-" in _:
        #         counter += 1
        # for _ in range(counter): 
        #     state_dict.update(torch.load(f"{dirname}/pytorch_model-0000{_+1}-of-0000{counter}.bin", map_location="cpu"))
        # load_msg = self.model.load_state_dict(
        #     state_dict,
        #     False,
        # )
        # print(load_mzsg)
        self.model.text_tokenizer.padding_side = "left"
        self.tokenizer = self.model.text_tokenizer
        self.patch_image_size = 336
        # self.mean = [0.481, 0.458, 0.408]
        # self.std = [0.269, 0.261, 0.276]
        # self.image_processor = transformers.CLIPImageProcessor()
        # self.image_processor = transforms.Compose(
        #         [
        #             transforms.Resize(
        #                 (self.patch_image_size, self.patch_image_size),
        #                 interpolation=transforms.InterpolationMode.BICUBIC,
        #             ),
        #             transforms.ToTensor(),
        #             transforms.Normalize(mean=self.mean, std=self.std),
        #         ]
        #     )
        self.patch_resize_transform = lambda x,y: process_image(x,y)
        self.model.eval()

    def generate(self, question: str, raw_image_data):
        input_data = get_pil_image(raw_image_data)
        if isinstance(input_data, Image.Image):
            if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
                vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(self.model.parameters()).dtype)
            else:
                # vision_x = self.image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
                try:
                    vision_x = self.patch_resize_transform(input_data,self.image_processor).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                except:
                    import pdb;pdb.set_trace()
                # import pdb;pdb.set_trace()
        else:
            raise ValueError("Invalid input data. Expected PIL Image.")

        lang_x = self.model.text_tokenizer(
            [
                get_formatted_prompt(question),
            ],
            return_tensors="pt",
        )

        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)
        lang_x_input_ids = lang_x["input_ids"]
        lang_x_attention_mask = lang_x["attention_mask"]
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x_input_ids.to(self.model.device),
            attention_mask=lang_x_attention_mask.to(self.model.device),
            max_new_tokens=128,
            num_beams=3,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id,
            early_stopping=True,
        )
        parsed_output = self.model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].split("<|endofchunk|>")[0].strip()
        return parsed_output

    def get_vision_x(self, input_data):
        if isinstance(input_data, Image.Image):
            if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
                vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(self.model.parameters()).dtype)
            else:
                vision_x = self.image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image.")
        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)
        return vision_x

    def eval_forward(self, question, answer, image):
        query = get_formatted_forward_prompt(question, answer)
        tokens = self.tokenizer(query, return_tensors="pt")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        with torch.no_grad():
            vision_x = self.get_vision_x(image)
            loss = self.model(vision_x=vision_x.to(self.model.device), lang_x=input_ids.to(self.model.device), attention_mask=attention_mask.to(self.model.device))[0]
        return loss


if __name__ == "__main__":
    model = OtterImage("/data/pufanyi/training_data/checkpoints/OTTER-Image-MPT7B")
    image = Image.open("/data/pufanyi/project/Otter-2/pipeline/evaluation/test_data/test.jpg")
    response = model.generate("What is this?", image)
    print(response)
    response = model.generate("What is this?", image)
    print(response)
