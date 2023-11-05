import os
import datetime
import json
import base64
from PIL import Image
import gradio as gr
import hashlib
import requests
from utils import build_logger
from conversation import model
import io


IMG_FLAG = "<image>"

LOGDIR = "log"
logger = build_logger("otter", LOGDIR)

current_model = model

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


def decode_image(encoded_image: str) -> Image:
    decoded_bytes = base64.b64decode(encoded_image.encode("utf-8"))
    buffer = io.BytesIO(decoded_bytes)
    image = Image.open(buffer)
    return image


def encode_image(image: Image.Image, format: str = "PNG") -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format=format)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_image


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_conv_image_dir():
    name = os.path.join(LOGDIR, "images")
    os.makedirs(name, exist_ok=True)
    return name


def get_image_name(image, image_dir=None):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    md5 = hashlib.md5(image_bytes).hexdigest()

    if image_dir is not None:
        image_name = os.path.join(image_dir, md5 + ".png")
    else:
        image_name = md5 + ".png"

    return image_name


def resize_image(image, max_size):
    width, height = image.size
    aspect_ratio = float(width) / float(height)

    if width > height:
        new_width = max_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height))
    return resized_image


def center_crop_image(image, max_aspect_ratio=1.5):
    width, height = image.size
    aspect_ratio = max(width, height) / min(width, height)

    if aspect_ratio >= max_aspect_ratio:
        if width > height:
            new_width = int(height * max_aspect_ratio)
            left = (width - new_width) // 2
            right = (width + new_width) // 2
            top = 0
            bottom = height
        else:
            new_height = int(width * max_aspect_ratio)
            left = 0
            right = width
            top = (height - new_height) // 2
            bottom = (height + new_height) // 2

        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image
    else:
        return image


def regenerate(dialog_state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    if dialog_state.messages[-1]["role"] == dialog_state.roles[1]:
        dialog_state.messages.pop()
    return (
        dialog_state,
        dialog_state.to_gradio_chatbot(),
    ) + (disable_btn,) * 4


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    dialog_state = current_model.copy()
    input_state = init_input_state()
    return (dialog_state, input_state, dialog_state.to_gradio_chatbot()) + (disable_btn,) * 4


def init_input_state():
    return {"images": [], "text": "", "images_ids": []}


def add_text(dialog_state, input_state, text, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}.")
    if text is None or len(text) == 0:
        return (dialog_state, input_state, "", dialog_state.to_gradio_chatbot()) + (no_change_btn,) * 4
    input_state["text"] += text

    if len(dialog_state.messages) > 0 and dialog_state.messages[-1]["role"] == dialog_state.roles[0]:
        dialog_state.messages[-1]["message"] = input_state
    else:
        dialog_state.messages.append({"role": dialog_state.roles[0], "message": input_state})
    print("add_text: ", dialog_state.to_gradio_chatbot())

    return (dialog_state, input_state, "", dialog_state.to_gradio_chatbot()) + (disable_btn,) * 4


def add_image(dialog_state, input_state, image, request: gr.Request):
    logger.info(f"add_image. ip: {request.client.host}.")
    if image is None:
        return (dialog_state, input_state, None, dialog_state.to_gradio_chatbot()) + (no_change_btn,) * 4

    image = image.convert("RGB")
    image = resize_image(image, max_size=224)
    image = center_crop_image(image, max_aspect_ratio=1.3)
    image_dir = get_conv_image_dir()
    image_path = get_image_name(image=image, image_dir=image_dir)
    if not os.path.exists(image_path):
        image.save(image_path)

    input_state["images"].append(image_path)
    input_state["text"]
    input_state["images_ids"].append(None)

    if len(dialog_state.messages) > 0 and dialog_state.messages[-1]["role"] == dialog_state.roles[0]:
        dialog_state.messages[-1]["message"] = input_state
    else:
        dialog_state.messages.append({"role": dialog_state.roles[0], "message": input_state})

    print("add_image:", dialog_state)

    return (dialog_state, input_state, None, dialog_state.to_gradio_chatbot()) + (disable_btn,) * 4


# def update_error_msg(chatbot, error_msg):
#     if len(error_msg) > 0:
#         info = '\n-------------\nSome errors occurred during response, please clear history and restart.\n' + '\n'.join(
#             error_msg)
#         chatbot[-1][-1] = chatbot[-1][-1] + info

#     return chatbot


def http_bot(image_input, text_input, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    print(f"Prompt request: {text_input}")

    base64_image_str = encode_image(image_input)

    payload = {
        "content": [
            {
                "prompt": text_input,
                "image": base64_image_str,
            }
        ],
        "token": "sk-OtterHD",
    }

    print(
        "request: ",
        {
            "prompt": text_input,
            "image": base64_image_str[:10],
        },
    )

    url = "http://10.128.0.40:8890/app/otter"
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    results = response.json()
    print("response: ", {"result": results["result"]})

    # output_state = init_input_state()
    # # image_dir = get_conv_image_dir()
    # output_state["text"] = results["result"]

    # for now otter doesn't have image output

    # for image_base64 in results['images']:
    #     if image_base64 == '':
    #         image_path = ''
    #     else:
    #         image = decode_image(image_base64)
    #         image = image.convert('RGB')
    #         image_path = get_image_name(image=image, image_dir=image_dir)
    #         if not os.path.exists(image_path):
    #             image.save(image_path)
    #     output_state['images'].append(image_path)
    #     output_state['images_ids'].append(None)

    # dialog_state.messages.append({"role": dialog_state.roles[1], "message": output_state})
    # # dialog_state.update_image_ids(results['images_ids'])

    # input_state = init_input_state()
    # chatbot = dialog_state.to_gradio_chatbot()
    # chatbot = update_error_msg(dialog_state.to_gradio_chatbot(), results['error_msg'])

    return results["result"]


def load_demo(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    dialog_state = current_model.copy()
    input_state = init_input_state()
    return dialog_state, input_state


title = """
# OTTER-HD: A High-Resolution Multi-modality Model
[[Otter Codebase]](https://github.com/Luodian/Otter) [[Paper]]() [[Checkpoints & Benchmarks]](https://huggingface.co/Otter-AI) 
         
"""

css = """
  #mkd {
    height: 1000px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

if __name__ == "__main__":
    with gr.Blocks(css=css) as demo:
        gr.Markdown(title)
        dialog_state = gr.State()
        input_state = gr.State()
        with gr.Tab("Ask a Question"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    image_input = gr.Image(label="Upload a High-Res Image", type="pil").style(height=600)
                with gr.Column(scale=1):
                    vqa_output = gr.Textbox(label="Output").style(height=600)
            text_input = gr.Textbox(label="Ask a Question")

            vqa_btn = gr.Button("Send It")

            gr.Examples(
                [
                    [
                        "/home/luodian/projects/Otter/archived/OtterHD/assets/G4_IMG_00095.png",
                        "How many camels are inside this image?",
                    ],
                    [
                        "/home/luodian/projects/Otter/archived/OtterHD/assets/G4_IMG_00095.png",
                        "How many people are inside this image?",
                    ],
                    [
                        "/home/luodian/projects/Otter/archived/OtterHD/assets/G4_IMG_00012.png",
                        "How many apples are there?",
                    ],
                    [
                        "/home/luodian/projects/Otter/archived/OtterHD/assets/G4_IMG_00080.png",
                        "What is this and where is it from?",
                    ],
                    [
                        "/home/luodian/projects/Otter/archived/OtterHD/assets/G4_IMG_00094.png",
                        "What's important on this website?",
                    ],
                ],
                inputs=[image_input, text_input],
                outputs=[vqa_output],
                fn=http_bot,
                label="Click on any Examples below👇",
            )
        vqa_btn.click(fn=http_bot, inputs=[image_input, text_input], outputs=vqa_output)

    demo.launch()
