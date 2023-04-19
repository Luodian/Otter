from PIL import Image
import requests
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from collie_core.collie_chat.chat import Chat, CONV_LANG, CONV_VISION
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--lm_path", required=True, help="path to language model")
    parser.add_argument("--checkpoint_path", help="path to best collie checkpoint, default to OpenFlamingo-9B")
    parser.add_argument("--cross_attn_every_n_layers", type=int, default=4, help="cross attention every n layers")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair " "in xxx=yyy format will be merged into config file (deprecate), " "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def initialize_model(lm_path, cross_attn_every_n_layers, checkpoint_path=None):
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14", clip_vision_encoder_pretrained="openai", lang_encoder_path=lm_path, tokenizer_path=lm_path, cross_attn_every_n_layers=cross_attn_every_n_layers
    )

    model.to("cuda")
    model.eval()

    if checkpoint_path is None:
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
    msg = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    print(msg)
    return model, image_processor, tokenizer


# ========================================
#             Model Initialization
# ========================================

args = parse_args()
model, image_processor, tokenizer = initialize_model(lm_path=args.lm_path, cross_attn_every_n_layers=args.cross_attn_every_n_layers, checkpoint_path=args.checkpoint_path)

# ========================================
#             Gradio Setting
# ========================================


def gradio_reset(chat_state):
    chat_state.messages = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder="Please upload your image first", interactive=False), gr.update(value="Upload & Start Chat", interactive=True), chat_state


def upload_and_anwer(image_1, image_2, image_3, text_1, text_2, text_3, num_beams):
    vision_x = [image_processor(image_1).unsqueeze(0), image_processor(image_2).unsqueeze(0), image_processor(image_3).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    vision_x = vision_x.to("cuda")
    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    lang_x = tokenizer(
        [f"<image>{text_1}<|endofchunk|><image>{text_2}<|endofchunk|><image>{text_3}"],
        return_tensors="pt",
    )
    lang_x = {k: v.to("cuda") for k, v in lang_x.items()}
    generated_tokens = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20,
        num_beams=num_beams,
    )
    generated_text = tokenizer.decode(generated_tokens[0])
    return generated_text


title = """<header>
<style>
h1 {text-align: center;}
</style>
<h1>Collie: A Visual Language Model with Efficient Instruction Tuning.</h1>
</header>
<section>
<h3>
    Collie interprets and deciphers complex visual information, enabling seamless integration of images and text. Collie is built on OpenFlamingo.
</h3>
<h3>
    It's currently under development and for internel test only.
</h3>
</section>"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    with gr.Row():
        with gr.Column(scale=0.5):
            image_1 = gr.Image(type="pil", value=Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw), label="Image_1")
            text_1 = gr.Textbox(label="text_1", placeholder="An image of two cats.")
            image_2 = gr.Image(type="pil", value=Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028137.jpg", stream=True).raw), label="Image_2")
            text_2 = gr.Textbox(label="text_2", placeholder="An image of a bathroom sink.")
            image_3 = gr.Image(type="pil", value=Image.open(requests.get("http://images.cocodataset.org/test-stuff2017/000000028352.jpg", stream=True).raw), label="Image_3")
            text_3 = gr.Textbox(label="text_3", placeholder="An image of")
            upload_button = gr.Button(value="Upload & Start", interactive=True, variant="primary")
            clear = gr.Button("Restart")

            with gr.Accordion("Advanced options", open=False):
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=3,
                    step=1,
                    interactive=True,
                    label="beam search numbers",
                )

        with gr.Column():
            output_text = gr.Textbox(label='User', placeholder='Ask me anything here')

            
    upload_button.click(
        upload_and_anwer,
        [image_1, image_2, image_3, text_1, text_2, text_3, num_beams],
        outputs=output_text,
    )

    # clear.click(gradio_reset, [chat_state], [chatbot, images, text_input, upload_button, chat_state], queue=False)

demo.launch(share=True, enable_queue=True)
