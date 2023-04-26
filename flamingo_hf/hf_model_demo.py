import requests
import torch
import transformers

# import sys
# sys.path.append(".")
from PIL import Image

from configuration_flamingo import FlamingoConfig
from modeling_flamingo import FlamingoModel


# config = FlamingoConfig.from_json_file("./flamingo_hf/config.json")
# model = FlamingoModel(config)

# # add <answer> token to tokenizer
# model.text_tokenizer.add_special_tokens(
#     {"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]}
# )

# model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))
# model.load_state_dict(torch.load("/home/v-boli7/azure_storage/models/multi_instruct_conversation-58k-nocontext_flamingo9B_lr1e-5/final_weight_hf.pt", map_location="cpu"), strict=False)
# model.save_pretrained("./flamingo_9b_hf")

from flamingo_hf import FlamingoModel

model = FlamingoModel.from_pretrained(
    "flamingo_9b_hf",
    device_map="auto",
    load_in_8bit=False,
)

tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()

demo_image_one = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
demo_image_two = Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028137.jpg", stream=True).raw)
query_image = Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True).raw)

vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
vision_x = vision_x.to(model.device)
"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
We also expect an <|endofchunk|> special token to indicate the end of the text 
portion associated with an image.
"""
model.text_tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
lang_x = model.text_tokenizer(
    ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
    return_tensors="pt",
)

"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", model.text_tokenizer.decode(generated_text[0]))