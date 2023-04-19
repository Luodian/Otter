from PIL import Image
import requests

from open_flamingo import create_model_and_transforms
# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="decapoda-research/llama-7b-hf",
    tokenizer_path="decapoda-research/llama-7b-hf",
    cross_attn_every_n_layers=4,
    # use_local_files=True,
)

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

"""
Step 1: Load images
"""
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
        stream=True
    ).raw
)

query_image = Image.open("/home/v-boli7/projects/PET-VLM/assets/psg_demo.png")
    # requests.get(
    #     "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
    #     stream=True
    # ).raw
# )

"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1 
 (this will always be one expect for video which we don't support yet), 
 channels = 3, height = 224, width = 224.
"""
vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>Output: Two cats sleeping.<|endofchunk|><image>Output: A bathroom sink.<|endofchunk|><image>Output:"],
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
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    no_repeat_ngram_size=1,
    prefix_allowed_tokens_fn=None,
    length_penalty=1.0,
    num_return_sequences=1,
    do_sample=True,
    early_stopping=True,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))