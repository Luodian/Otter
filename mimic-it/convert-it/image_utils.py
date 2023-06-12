import base64
import os
import cv2

from io import BytesIO
from PIL import Image
from typing import Dict


def resize_image(image, target_size=(224, 224)):
    """
    Resizes the given image to the target size using the Lanczos algorithm.

    Args:
        image (PIL.Image.Image): The input image to be resized.
        target_size (tuple[int, int]): The target size to which the image should be resized.
            Defaults to (224, 224).

    Returns:
        PIL.Image.Image: The resized image.
    """
    if image.size != target_size:
        return image.resize(target_size, Image.LANCZOS)
    return image


def process_image(img: Image):
    """
    Processes the input image by resizing it, converting it to RGB mode, and encoding it as base64.

    Args:
        img (PIL.Image.Image): The input image to be processed.

    Returns:
        str: The base64 encoded string representation of the processed image.
    """
    resized_img = resize_image(img)
    if resized_img.mode != "RGB":
        resized_img = resized_img.convert("RGB")
    buffer = BytesIO()
    resized_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64


def get_json_data(images: Dict[str, Image], dataset_name: str) -> Dict[str, str]:
    """
    Converts a dictionary of images to a JSON-compatible dictionary with base64 encoded strings.

    Args:
        images (Dict[str, Image]): A dictionary of images, where the keys are image identifiers and the values are PIL.Image.Image objects.
        dataset_name (str): The name of the dataset.

    Returns:
        Dict[str, str]: A dictionary where the keys are formatted as "{dataset_name}_IMG_{key}" and the values are base64 encoded string representations of the processed images.
    """
    json_data = {}
    for key, img in images.items():
        new_key = f"{dataset_name}_IMG_{key}"
        json_data[new_key] = process_image(img)
    return json_data


def save_video_to_b64(video_file, fps=1):
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file {video_file} does not exist.")

    cap = cv2.VideoCapture(video_file)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    saved_frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % (video_fps // fps) == 0:
            # convert frame to base64
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))
            saved_frame_count += 1

        frame_count += 1

    cap.release()

    return frames
