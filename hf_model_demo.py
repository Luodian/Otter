import requests
import torch
import transformers

# import sys
# sys.path.append(".")
from PIL import Image

from flamingo_hf.configuration_flamingo import FlamingoConfig
from flamingo_hf.modeling_flamingo import FlamingoModel
from flamingo_hf import FlamingoForConditionalGeneration

config = FlamingoConfig.from_json_file("./flamingo_hf/config.json")
# model = FlamingoForConditionalGeneration(config)

# model.load_state_dict(torch.load("/home/v-boli7/azure_storage/models/openflamingo/open_flamingo_hf.pt", map_location="cpu"), strict=False)
# model = FlamingoForConditionalGeneration.from_pretrained(
#     pretrained_model_name_or_path="/home/v-boli7/projects/openflamingo-9b-hf",
#     device_map="auto",
# )

correct_model = FlamingoForConditionalGeneration(config)
correct_model.load_state_dict(torch.load("/home/v-boli7/azure_storage/models/openflamingo/open_flamingo_hf.pt", map_location="cpu"), strict=False)

# import copy
# model.lang_encoder.lm_head.weight = copy.deepcopy(correct_model.lang_encoder.lm_head.weight)

tokenizer = correct_model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()

demo_image_one = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
demo_image_two = Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028137.jpg", stream=True).raw)
query_image = Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True).raw)

vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
We also expect an <|endofchunk|> special token to indicate the end of the text 
portion associated with an image.
"""
correct_model.text_tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
lang_x = correct_model.text_tokenizer(
    ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
    return_tensors="pt",
)

"""
Step 4: Generate text
"""
generated_text = correct_model.generate(
    vision_x=vision_x.to(correct_model.device),
    lang_x=lang_x["input_ids"].to(correct_model.device),
    attention_mask=lang_x["attention_mask"].to(correct_model.device),
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", correct_model.text_tokenizer.decode(generated_text[0]))