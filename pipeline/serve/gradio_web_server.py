import argparse
from collections import defaultdict
import datetime
import json
import os
import time
import uuid
import gradio as gr
import requests
import re
from pipeline.conversation import default_conversation, conv_templates, SeparatorStyle
from pipeline.constants import LOGDIR
from pipeline.serve.serving_utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
)
from pipeline.serve.gradio_patch import Chatbot as grChatbot
from pipeline.serve.gradio_css import code_highlight_css

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_DEMO_END_TOKEN = "<|endofchunk|>"
# DEFAULT_ANSWER_TOKEN = "<answer>"
template_name = "otter"

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "Open Flamingo Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

priority = {
    "otter": "aaaaaaa",
    "open_flamingo": "aaaaaab",
    "otter_video": "aaaaaac",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(value=model, visible=True)

    state = None
    return (
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    return (
        state,
        gr.Dropdown.update(choices=models, value=models[0] if len(models) > 0 else ""),
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    state.skip_next = False
    return (
        (
            state,
            state.to_gradio_chatbot(),
        )
        + (
            "",
            "",
            None,
        )
        * 2
        + (
            "",
            None,
        )
        * 1
        + (disable_btn,) * 5
    )


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = None
    return (
        (
            state,
            [],
        )
        + (
            "",
            "",
            None,
        )
        * 2
        + (
            "",
            None,
        )
        * 1
        + (disable_btn,) * 5
    )


def add_text(
    state,
    model_selector,
    text_demo_question_1,
    text_demo_answer_1,
    image_demo_1,
    text_demo_question_2,
    text_demo_answer_2,
    image_demo_2,
    text_3,
    image_3,
    request: gr.Request,
):
    if text_demo_question_1 != "":
        text_demo_question_1 = text_demo_question_1.strip()
        if not re.search(r"[.,?]$", text_demo_question_1):
            text_demo_question_1 += "."
    if text_demo_answer_2 != "":
        text_demo_question_2 = text_demo_question_2.strip()
        if not re.search(r"[.,?]$", text_demo_question_1):
            text_demo_question_1 += "."
    if text_3 != "":
        text_3 = text_3.strip()
        if not re.search(r"[.,?]$", text_3):
            text_3 += "."
    template_name = "otter" if "otter" in model_selector.lower() else "open_flamingo"
    if "otter" in model_selector.lower():
        DEFAULT_ANSWER_TOKEN = "<answer> "
        human_role_label = conv_templates[template_name].copy().roles[0] + ": "
        bot_role_label = " " + conv_templates[template_name].copy().roles[1] + ":"
    else:
        DEFAULT_ANSWER_TOKEN = ""
        human_role_label = ""
        bot_role_label = ""
    text = text_3
    if conv_templates[template_name].copy().roles[1] is not None:
        text += " " + conv_templates[template_name].copy().roles[1] + ":" + DEFAULT_ANSWER_TOKEN
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if state is None:
        state = conv_templates[template_name].copy()
        logger.info(f"TEMPLATE. {state}")
    if len(text) <= 0 and image_3 is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"violate moderation. ip: {request.client.host}. text: {text}")
            state.skip_next = True
            return (
                (state, state.to_gradio_chatbot())
                + (
                    "",
                    "",
                    None,
                )
                * 2
                + (moderation_msg, None)
                + (disable_btn,) * 5
            )

    text = text[:1536]  # Hard cut-off

    text = human_role_label + text
    if image_3 is not None:
        text = DEFAULT_IMAGE_TOKEN + text
    if image_3 is None and len(state.messages) >= 2:
        # text = DEFAULT_IMAGE_TOKEN + text
        image_3 = state.messages[-2][1][3]

    # # clean state if it's a new conversation
    # if image_3 is not None and state is not None:
    #     state = conv_templates[template_name].copy()
    #     logger.info(f"TEMPLATE. {state}")

    if text_demo_answer_2 != "":
        assert image_demo_2 is not None
        text = (
            DEFAULT_IMAGE_TOKEN
            + human_role_label
            + text_demo_question_2
            + bot_role_label
            + DEFAULT_ANSWER_TOKEN
            + text_demo_answer_2
            + DEFAULT_DEMO_END_TOKEN
            + text
        )
    if text_demo_answer_1 != "":
        assert image_demo_1 is not None
        text = (
            DEFAULT_IMAGE_TOKEN
            + human_role_label
            + text_demo_question_1
            + bot_role_label
            + DEFAULT_ANSWER_TOKEN
            + text_demo_answer_1
            + DEFAULT_DEMO_END_TOKEN
            + text
        )

    input = (text, image_demo_1, image_demo_2, image_3)
    state.append_message(state.roles[0], input)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (
        (
            state,
            state.to_gradio_chatbot(),
        )
        + (
            "",
            "",
            None,
        )
        * 2
        + (
            "",
            None,
        )
        * 1
        + (disable_btn,) * 5
    )


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def http_bot(
    state,
    model_selector,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    no_repeat_ngram_size,
    length_penalty,
    do_sample,
    early_stopping,
    request: gr.Request,
):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector
    template_name = "otter" if "otter" in model_selector.lower() else "open_flamingo"

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        new_state = conv_templates[template_name].copy()
        new_state.conv_id = uuid.uuid4().hex
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (
            state,
            state.to_gradio_chatbot(),
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    # Construct prompt
    prompt = state.get_prompt()
    prompt = prompt.strip()
    # if state.roles[1] is not None:
    #     role_label = state.roles[1] + ": "
    #     # hard code preprocessing: remove the last role label
    #     prompt = prompt[: -len(role_label)]

    # Construct generation kwargs
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "length_penalty": length_penalty,
        "do_sample": do_sample,
        "early_stopping": early_stopping,
    }

    # Make requests

    pload = {
        "model": model_name,
        "prompt": prompt,
        "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else state.sep2,
        "images": f"List of {len(state.get_images())} images",
        "generation_kwargs": generation_kwargs,
    }
    logger.info(f"==== request ====\n{pload}")

    pload["images"] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=25,
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    # output = data["text"][len(prompt) + 1 :].strip() # original postprocessing
                    output = data["text"].strip()  # TODO: fix hardcode postprocessing
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


title_markdown = """
<header>
<style>
h1 {text-align: center;}
a:link {
  text-decoration: none;
}
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}
</style>
<h1><a href="https://github.com/Luodian/otter"><img src="https://i.postimg.cc/MKmyP9wH/new-banner.png" alt="Otter: Multi-Modal In-Context Learning Model with Instruction Tuning" width="500px" class="center"></a></h1>
</header>
<h2><a href="https://github.com/Luodian/otter"><img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" style="height: 15px; display:inline;" class="icon" alt="github">GitHub</a>
&nbsp;&nbsp;&nbsp;
<a href="https://youtu.be/K8o_LKGQJhs"><img src="https://www.svgrepo.com/show/13671/youtube.svg" style="height: 15px; display:inline;" class="icon" alt="video demo">Video</a>
&nbsp;&nbsp;&nbsp;
<a href="https://ottervideo.cliangyu.com/"><img src="https://www.svgrepo.com/show/2065/chat.svg" style="height: 15px; display:inline;" class="icon" alt="live demo">Live Demo (Video Otter)</a>
&nbsp;&nbsp;&nbsp;
<img style="height: 20px; display:inline;" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fotter.cliangyu.com&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false"/>

</h2>

<span style="font-size:larger;">

### Note:
Current Otter Image is version **Otter-MPT7B (0710/0713)**. We update our models by incoporating OpenFlamingv2 and specifically tune it to enable generation abilities for both long and short answers.

This version Otter Image demonstrates in-context learning ability to demonstrate more reasonable and coherent answer following given example instruction/response pairs. 
</span>

We currently **dont support language-only chat** (the model could but our code doesnt allow it for now). Since we aim to demonstrate the ability of chatting on images, you may need to upload your images first and then ask it questions.

Otter can read multiple images and answer multiple questions towards the same image (visually the image will appear in chatbox again due to our implementation).

If you find it's interesting, please consider to star our [github](https://github.com/Luodian/Otter) and cite our [paper](https://arxiv.org/abs/2306.05425). What we do is all to make the community better and to approach the goal of AI for helping people's life.

Sometimes we are experiencing server overload, and as the model is hosted on a dual-RTX-3090 machine. Please try again later if you encounter any error or contact drluodian@gmail.com for any problem.

Sometimes the model may behave weirdly if you didnt clear the chat history. Please do clear the chat history to make sure Otter reads your inputs correctly.
"""

tos_markdown = """
### Terms of Use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator. For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
"""


learn_more_markdown = """
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA. Please contact us if you find any potential violation.
"""


css = (
    code_highlight_css
    + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""
)


def build_demo(embed_mode):
    with gr.Blocks(title="Otter Chat", theme=gr.themes.Base(), css=css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                    ).style(container=False)

                imagebox_3 = gr.Image(label="Image", type="pil")

                with gr.Row():
                    imagebox_demo_1 = gr.Image(label="Demo Image 1 (optional)", type="pil")
                    textbox_demo_question_1 = gr.Textbox(
                        label="Demo Text Query 1 (optional)",
                        show_label=True,
                        placeholder="Example: What is in the image?",
                    ).style(container=True)
                    textbox_demo_answer_1 = gr.Textbox(
                        label="Demo Text Answer 1 (optional)",
                        show_label=True,
                        placeholder="<Describe Demo Image 1>",
                    ).style(container=True)
                with gr.Row():
                    imagebox_demo_2 = gr.Image(label="Demo Image 2 (optional)", type="pil")
                    textbox_demo_question_2 = gr.Textbox(
                        label="Demo Text Query 2 (optional)",
                        show_label=True,
                        placeholder="Example: What is in the image?",
                    ).style(container=True)
                    textbox_demo_answer_2 = gr.Textbox(
                        label="Demo Text Answer 2 (optional)",
                        show_label=True,
                        placeholder="<Describe Demo Image 2>",
                    ).style(container=True)

                with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
                    max_new_tokens = gr.Slider(minimum=16, maximum=512, value=512, step=1, interactive=True, label="# generation tokens")
                    temperature = gr.Slider(minimum=0, maximum=1, value=1, step=0.1, interactive=True, label="temperature")
                    top_k = gr.Slider(minimum=0, maximum=10, value=0, step=1, interactive=True, label="top_k")
                    top_p = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.1, interactive=True, label="top_p")
                    no_repeat_ngram_size = gr.Slider(minimum=1, maximum=10, value=3, step=1, interactive=True, label="no_repeat_ngram_size")
                    length_penalty = gr.Slider(minimum=1, maximum=5, value=1, step=0.1, interactive=True, label="length_penalty")
                    do_sample = gr.Checkbox(interactive=True, label="do_sample")
                    early_stopping = gr.Checkbox(interactive=True, label="early_stopping", value=True)

            with gr.Column(scale=6):
                chatbot = grChatbot(elem_id="chatbot", visible=False).style(height=720)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox_3 = gr.Textbox(
                            label="Text Query",
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                        ).style(container=False)
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=False)
                with gr.Row(visible=False) as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gr.Examples(
            label="Examples (0-shot)",
            examples=[
                [f"{cur_dir}/examples/ms_st.jpg", "Does the image feature a globally recognized technology company?"],
                [f"{cur_dir}/examples/ms_st.jpg", "Does the image feature a globally recognized technology company? Please answer with yes or no."],
                [f"{cur_dir}/examples/zelda_princess.jpg", "Can you identify the game character?"],
                [f"{cur_dir}/examples/martin.jpeg", "Can you identify the historic figure?"],
                [f"{cur_dir}/examples/gtav.jpg", "Can you identify what the image is about?"],
                [
                    f"{cur_dir}/examples/xray.jpg",
                    "Act as a radiologist and write a diagnostic radiology report for the patient based on their chest radiographs:",
                ],
                [
                    f"{cur_dir}/examples/baseball.jpg",
                    "Please describe this image in short words.",
                ],
                [
                    f"{cur_dir}/examples/waterview.jpg",
                    "Please provide a detailed description of the image and share your personal impressions of the scene.",
                ],
            ],
            inputs=[
                imagebox_3,
                textbox_3,
            ],
        )
        gr.Examples(
            label="In-Context Examples (1-shot)",
            examples=[
                [
                    f"{cur_dir}/examples/think_different.png",
                    "What's written on this image? Please answer in short.",
                    f"{cur_dir}/examples/pepsi.png",
                    "What's written on this image? Please answer in short.",
                    "pepsi, is pepsi ok?",
                    # f"{cur_dir}/examples/subway.png",
                    # "What's written on this image? Please answer in short.",
                    # "SUBWAY, eat fresh",
                ],
                [
                    f"{cur_dir}/examples/dinner.jpg",
                    "An image of",
                    f"{cur_dir}/examples/cat.jpg",
                    "An image of",
                    "two cats.",
                    # f"{cur_dir}/examples/bathroom.jpg",
                    # "An image of",
                    # "a bathroom sink.",
                ],
            ],
            inputs=[
                imagebox_3,
                textbox_3,
                imagebox_demo_1,
                textbox_demo_question_1,
                textbox_demo_answer_1,
                imagebox_demo_2,
                # textbox_demo_question_2,
                # textbox_demo_answer_2,
            ],
        )

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        demo_list = [
            textbox_demo_question_1,
            textbox_demo_answer_1,
            imagebox_demo_1,
            textbox_demo_question_2,
            textbox_demo_answer_2,
            imagebox_demo_2,
        ]
        prarameter_list = [
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            no_repeat_ngram_size,
            length_penalty,
            do_sample,
            early_stopping,
        ]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox_3, upvote_btn, downvote_btn, flag_btn],
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox_3, upvote_btn, downvote_btn, flag_btn],
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox_3, upvote_btn, downvote_btn, flag_btn],
        )
        regenerate_btn.click(
            regenerate,
            state,
            [
                state,
                chatbot,
            ]
            + demo_list
            + [
                textbox_3,
                imagebox_3,
            ]
            + btn_list,
        ).then(
            http_bot,
            [
                state,
                model_selector,
            ]
            + prarameter_list,
            [state, chatbot] + btn_list,
        )
        clear_btn.click(
            clear_history,
            None,
            [
                state,
                chatbot,
            ]
            + demo_list
            + [
                textbox_3,
                imagebox_3,
            ]
            + btn_list,
        )

        textbox_3.submit(
            add_text,
            [
                state,
                model_selector,
            ]
            + demo_list
            + [
                textbox_3,
                imagebox_3,
            ],
            [
                state,
                chatbot,
            ]
            + demo_list
            + [
                textbox_3,
                imagebox_3,
            ]
            + btn_list,
        ).then(
            http_bot,
            [
                state,
                model_selector,
            ]
            + prarameter_list,
            [state, chatbot] + btn_list,
        )
        submit_btn.click(
            add_text,
            [
                state,
                model_selector,
            ]
            + demo_list
            + [
                textbox_3,
                imagebox_3,
            ],
            [
                state,
                chatbot,
            ]
            + demo_list
            + [
                textbox_3,
                imagebox_3,
            ]
            + btn_list,
        ).then(
            http_bot,
            [
                state,
                model_selector,
            ]
            + prarameter_list,
            [state, chatbot] + btn_list,
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [
                    state,
                    model_selector,
                    chatbot,
                    textbox_3,
                    submit_btn,
                    button_row,
                    parameter_row,
                ],
                _js=get_window_url_params,
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [
                    state,
                    model_selector,
                    chatbot,
                    textbox_3,
                    submit_btn,
                    button_row,
                    parameter_row,
                ],
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    gr.close_all()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default="7861")
    parser.add_argument("--controller_url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency_count", type=int, default=16)
    parser.add_argument("--model_list_mode", type=str, default="reload", choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")
    models = get_model_list()
    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )
    gr.close_all()
