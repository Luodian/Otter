import sys
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, FuyuImageProcessor, CLIPImageProcessor, IdeficsForVisionText2Text, FuyuImageProcessor

# from src.otter_ai.models.fuyu.modeling_fuyu import FuyuForCausalLM
from transformers import FuyuForCausalLM
from src.otter_ai.models.fuyu.processing_fuyu import FuyuProcessor
from otter_ai import OtterForConditionalGeneration
import io
import base64

sys.path.append("../..")
from pipeline.train.train_utils import get_image_attention_mask

requests.packages.urllib3.disable_warnings()


def get_pil_image(raw_image_data) -> Image.Image:
    if isinstance(raw_image_data, Image.Image):
        return raw_image_data

    elif isinstance(raw_image_data, dict) and "bytes" in raw_image_data:
        return Image.open(io.BytesIO(raw_image_data["bytes"]))

    elif isinstance(raw_image_data, str):  # Assuming this is a base64 encoded string
        image_bytes = base64.b64decode(raw_image_data)
        return Image.open(io.BytesIO(image_bytes))

    else:
        raise ValueError("Unsupported image data format")


class TestOtter:
    def __init__(self, checkpoint) -> None:
        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        self.model = OtterForConditionalGeneration.from_pretrained(checkpoint, **kwargs)
        self.image_processor = CLIPImageProcessor()
        self.tokenizer = self.model.text_tokenizer
        self.tokenizer.padding_side = "left"
        self.model.eval()

    def generate(self, image, prompt, no_image_flag=False):
        input_data = image
        if isinstance(input_data, Image.Image):
            if no_image_flag:
                vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(self.model.parameters()).dtype)
            else:
                vision_x = self.image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image.")

        lang_x = self.tokenizer(
            [
                self.get_formatted_prompt(prompt, no_image_flag=no_image_flag),
            ],
            return_tensors="pt",
        )

        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)

        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output = self.tokenizer.decode(generated_text[0]).split("<answer>")[-1].strip().replace("<|endofchunk|>", "")
        return output

    def get_formatted_prompt(self, question: str, no_image_flag: str) -> str:
        if no_image_flag:
            return f"User:{question} GPT:<answer>"
        else:
            return f"<image>User:{question} GPT:<answer>"


class TestIdefics:
    def __init__(self, checkpoint: str = "HuggingFaceM4/idefics-9b-instruct"):
        self.model = IdeficsForVisionText2Text.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.model.eval()

    def generate(self, image, prompt, no_image_flag=False):
        input_data = image
        if isinstance(input_data, Image.Image):
            if no_image_flag:
                vision_x = torch.zeros(1, 1, 3, 224, 224, dtype=next(self.model.parameters()).dtype)
            else:
                vision_x = self.image_processor.preprocess([input_data], return_tensors="pt").unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image.")

        lang_x = self.tokenizer(
            [
                self.get_formatted_prompt(prompt, no_image_flag=no_image_flag),
            ],
            return_tensors="pt",
        )

        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)
        lang_x = self.tokenizer(
            [
                self.get_formatted_prompt(prompt, no_image_flag=no_image_flag),
            ],
            return_tensors="pt",
        )
        image_attention_mask = get_image_attention_mask(lang_x["input_ids"], 1, self.tokenizer, include_image=not no_image_flag)
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>", "User:"], add_special_tokens=False).input_ids
        generated_ids = self.model.generate(
            pixel_values=vision_x.to(self.model.device),
            input_ids=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            image_attention_mask=image_attention_mask.to(self.model.device),
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            top_p=0.5,
        )
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        output = output.split("Assistant:")[1].strip().replace("<end_of_utterance>", "")
        return output

    def get_formatted_prompt(self, question: str, no_image_flag: str) -> str:
        if no_image_flag:
            return f"User:{question}<end_of_utterance>\nAssistant:"
        else:
            return f"User:<fake_token_around_image><image><fake_token_around_image>{question}<end_of_utterance>\nAssistant:"


class TestOtterHD:
    def __init__(self, checkpoint: str = "adept/fuyu-8b", cuda_id: int = 0, resolution: int = 336, max_new_tokens=256):
        self.resolution = resolution
        self.device = f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
        self.model = FuyuForCausalLM.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("adept/fuyu-8b")
        self.image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(image_processor=self.image_processor, tokenizer=self.tokenizer)
        self.max_new_tokens = max_new_tokens
        self.bad_words_list = ["User:", "Assistant:"]
        self.bad_words_ids = self.tokenizer(self.bad_words_list, add_special_tokens=False).input_ids

    def generate(self, image, prompt, no_image_flag=False):
        raw_image_data = get_pil_image(image)
        # make sure the image is in RGB format and resize to match the width
        # max_height, max_width = self.resolution, self.resolution
        raw_image_data = raw_image_data.convert("RGB")
        if max(raw_image_data.size) > 1080:
            raw_image_data.thumbnail((1080, 1080), Image.ANTIALIAS)

        print(f"Eval with res: {raw_image_data.size}")
        # raw_image_data.thumbnail((max_width, max_height), Image.ANTIALIAS)
        formated_prompt = f"User: {prompt} Assistant:"
        model_inputs = self.processor(text=formated_prompt, images=[raw_image_data] if no_image_flag is False else None, device=self.device)
        for k, v in model_inputs.items():
            model_inputs[k] = v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else [vv.to(self.device, non_blocking=True) for vv in v]

        generation_output = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
        return generation_text[0].split("\x04")[1].strip(" ").strip("\n")
