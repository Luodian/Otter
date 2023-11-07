from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import AutoTokenizer, FuyuForCausalLM, FuyuProcessor, FuyuImageProcessor
import base64
import re
from io import BytesIO
from datetime import datetime
import hashlib
from PIL import Image
import io, os

app = Flask(__name__)

# Initialization code (similar to what you have in your Gradio demo)
model_id = input("Model ID: ")
device = "cuda:0"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
processor = FuyuProcessor(image_processor=FuyuImageProcessor(), tokenizer=tokenizer)

# Ensure model is in evaluation mode
model.eval()
prompt_txt_path = "../user_logs/prompts.txt"
images_folder_path = "../user_logs"


def save_image_unique(pil_image, directory=images_folder_path):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Convert the PIL Image into a bytes object
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Compute the hash of the image data
    hasher = hashlib.sha256()
    hasher.update(img_byte_arr)
    hash_hex = hasher.hexdigest()

    # Create a file name with the hash value
    file_name = f"{hash_hex}.png"
    file_path = os.path.join(directory, file_name)

    # Check if a file with the same name exists
    if os.path.isfile(file_path):
        print(f"Image already exists with the name: {file_name}")
    else:
        # If the file does not exist, save the image
        with open(file_path, "wb") as new_file:
            new_file.write(img_byte_arr)
        print(f"Image saved with the name: {file_name}")

    return file_path


# Define endpoint
@app.route("/app/otter", methods=["POST"])
def process_image_and_prompt():
    start_time = datetime.now()
    # Parse request data
    data = request.get_json()
    query_content = data["content"][0]
    if "image" not in query_content:
        return jsonify({"error": "Missing Image"}), 400
    elif "prompt" not in query_content:
        return jsonify({"error": "Missing Prompt"}), 400

    # Decode the image
    image_data = query_content["image"]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    prompt = query_content["prompt"]
    formated_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    image_path = save_image_unique(image)

    # Preprocess the image and prompt, and run the model
    response = predict(image, prompt)
    torch.cuda.empty_cache()

    with open(prompt_txt_path, "a") as f:
        f.write(f"*************************{formated_time}**************************" + "\n")
        f.write(f"Image saved to {image_path}" + "\n")
        f.write(f"Prompt: {prompt}" + "\n")
        f.write(f"Response: {response}" + "\n\n")

    # Return the response
    return jsonify({"result": response})


import time


# Other necessary functions (adapted from your Gradio demo)
def predict(image, prompt):
    time_start = time.time()
    image = image.convert("RGB")
    # if max(image.size) > 1080:
    #     image.thumbnail((1080, 1080))
    model_inputs = processor(text=prompt, images=[image], device=device)
    for k, v in model_inputs.items():
        model_inputs[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else [vv.to(device, non_blocking=True) for vv in v]
    model_inputs["image_patches"][0] = model_inputs["image_patches"][0].to(dtype=next(model.parameters()).dtype)

    generation_output = model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id=processor.tokenizer.eos_token_id
        # do_sample=True,
        # top_p=0.5,
        # temperature=0.2,
    )
    generation_text = processor.batch_decode(generation_output, skip_special_tokens=True)
    generation_text = [text.split("\x04")[1].strip() for text in generation_text]
    end_time = time.time()
    formated_interval = f"{end_time - time_start:.3f}"
    response = f"Image Resolution (W, H): {image.size}\n-------------------\nModel Respond Time(s): {formated_interval}\n-------------------\nAnswer: {generation_text[0]}"
    return response


# Utility functions (as per the Gradio script, you can adapt the same or similar ones)
# ... (e.g., resize_to_max, pad_to_size, etc.)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8890)
