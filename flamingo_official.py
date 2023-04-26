from open_flamingo.src.flamingo_hf import FlamingoPreTrainedModel
from PIL import Image
import requests

model_kwargs = {"device_map": "auto", "load_in_8bit": True}
model = FlamingoPreTrainedModel.from_pretrained(
    "/media/ntu/volume2/s121md302_06/code/mutoo/PET-VLM/checkpoint/openflamingo_hf",
    # **model_kwargs
)

"""
Step 1: Load images
"""
demo_image_one = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

demo_image_two = Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028137.jpg", stream=True).raw)

query_image = Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True).raw)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
batch_size x num_media x num_frames x channels x height x width. 
In this case batch_size = 1, num_media = 3, num_frames = 1 
(this will always be one expect for video which we don't support yet), 
channels = 3, height = 224, width = 224.
"""
import torch

vision_x = [model.image_processor(demo_image_one).unsqueeze(0), model.image_processor(demo_image_two).unsqueeze(0), model.image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

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
