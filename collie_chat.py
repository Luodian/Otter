import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from collie_core.collie_chat.chat import Chat, CONV_LANG, CONV_VISION
# from minigpt4.common.config import Config
# from minigpt4.common.dist_utils import get_rank
# from minigpt4.common.registry import registry
# from minigpt4.conversation.conversation import Chat, CONV_VISION

# # imports modules for registration
# from minigpt4.datasets.builders import *
# from minigpt4.models import *
# from minigpt4.processors import *
# from minigpt4.runners import *
# from minigpt4.tasks import *

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
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
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=lm_path,
        tokenizer_path=lm_path,
        cross_attn_every_n_layers=cross_attn_every_n_layers
    )

    model.to("cuda")
    model.eval()

    # grab model checkpoint from huggingface hub
    # checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
    if checkpoint_path is not None:
        msg = model.load_state_dict(torch.load(checkpoint_path), strict=False)
        print(msg)
    return model, image_processor, tokenizer

# ========================================
#             Model Initialization
# ========================================

model, image_processor, tokenizer = initialize_model(lm_path="facebook/opt-1.3b", cross_attn_every_n_layers=1)
chat_model = Chat(model, image_processor, tokenizer)

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state):
    chat_state.messages = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state

def upload_img(gr_img, input_text, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True)
    chat_state = CONV_VISION.copy()
    llm_message = chat_model.upload_img(gr_img, chat_state)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state

def gradio_ask(user_message, chatbot, chat_state=None):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat_state = CONV_LANG.copy()
    chat_model.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, num_beams, temperature):
    llm_message = chat_model.answer(conv=chat_state, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state


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
# description = """<h3>This is the demo of Collie. Upload your images and start chatting!</h3>"""
# article = """<strong>Paper</strong>: <a href='https://github.com/Vision-CAIR/MiniGPT-4/blob/main/MiniGPT_4.pdf' target='_blank'>Here</a>
# <strong>Code</strong>: <a href='https://github.com/Vision-CAIR/MiniGPT-4' target='_blank'>Here</a>
# <strong>Project Page</strong>: <a href='https://minigpt-4.github.io/' target='_blank'>Here</a>
# """

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil", value="./assets/demo1.jpg", label="Image")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
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
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="temperature",
                )

        with gr.Column():
            chat_state = gr.State()
            chatbot = gr.Chatbot(label='Collie')
            text_input = gr.Textbox(label='User', placeholder='Ask me anything here')
    
    upload_button.click(upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, num_beams, temperature], [chatbot, chat_state]
    )
    clear.click(gradio_reset, [chat_state], [chatbot, image, text_input, upload_button, chat_state], queue=False)

demo.launch(share=True, enable_queue=True)