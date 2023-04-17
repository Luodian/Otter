import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

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


# ========================================
#             Model Initialization
# ========================================

# print('Initializing Chat')
# cfg = Config(parse_args())

# model_config = cfg.model_cfg
# model_cls = registry.get_model_class(model_config.arch)
# model = model_cls.from_config(model_config).to('cuda:0')

# vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
# vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
# chat = Chat(model, vis_processor)
# print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    chat_state.messages = []
    img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True)
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list

title = """<header>
                <h1>Welcome to Collie Chat!</h1>
                <img src="./assets/collie_icon.png" alt="Collie Icon">
            </header>
            <section>
                <h3>
                    Collie is a visual language models tuned by instruction following. Collie interprets and deciphers complex visual information, enabling seamless integration of images and text. Collie is built on OpenFlamingo.
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
    # gr.Markdown(description)
    # gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=16,
                value=5,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='Collie')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)
    
    upload_button.click(upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, upload_button, chat_state, img_list], queue=False)

demo.launch(share=True, enable_queue=True)