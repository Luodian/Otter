import requests
import torch
import transformers
import json
from PIL import Image
from ..otter.modeling_otter import OtterForConditionalGeneration


model = OtterForConditionalGeneration.from_pretrained(
    "luodian/otter-9b-hf", device_map="auto"
)
model.text_tokenizer.padding_side = "left"
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()


def get_image(url: str) -> Image.Image:
    """
    Get image from url
    
    Args:
        url (str): url of the image
    
    Returns:
        Image.Image: PIL Image
    """
    return Image.open(requests.get(url, stream=True).raw)


def get_formatted_prompt(prompt: str) -> str:
    """
    Format prompt for GPT
    
    Args:
        prompt (str): prompt to be formatted
    
    Returns:
        str: formatted prompt
    """
    return f"<image>User: {prompt} GPT:<answer>"


def get_response(url: str, prompt: str) -> str:
    """
    Get the response of single image and prompt from the model
    
    Args:
        url (str): url of the image
        prompt (str): the prompt (no need to be formatted)
    
    Returns:
        str: response of the model
    """
    query_image = get_image(url)
    vision_x = (
        image_processor.preprocess(
            [query_image], return_tensors="pt"
        )["pixel_values"]
        .unsqueeze(1)
        .unsqueeze(0)
    )
    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x["input_ids"].to(model.device),
        attention_mask=lang_x["attention_mask"].to(model.device),
        max_new_tokens=256,
        num_beams=1,
        no_repeat_ngram_size=3,
    )
    return model.text_tokenizer.decode(generated_text[0])


if __name__ == "__main__":
    responses = []
    with open("benchmark.json") as f:
        data = json.load(f)
        for item in data:
            responses.append({
                "image": item["image"],
                "prompt": item["prompt"],
                "response": get_response(item["image"], item["prompt"])
            })
    json.dump(responses, open("benchmark_responses.json", "w"), indent=4)
