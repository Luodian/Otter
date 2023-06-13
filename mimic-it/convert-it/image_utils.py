import base64
import os
import cv2

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def get_image_id(image_name: str, dataset_name: str) -> str:
    """
    Extracts the image identifier from a given image path.

    Args:
        image_path (str): The path to the image.
        dataset_name (str): The name of the dataset.

    Returns:
        str: The image identifier.
    """
    return f"{dataset_name}_IMG_{get_image_name(image_name)}"


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


def get_json_data(
    images: dict[str, Image], dataset_name: str, num_thread: int
) -> dict[str, str]:
    """
    Converts a dictionary of images to a JSON-compatible dictionary with base64 encoded strings.

    Args:
        images (Dict[str, Image]): A dictionary of images, where the keys are image identifiers and the values are PIL.Image.Image objects.
        dataset_name (str): The name of the dataset.
        num_thread (int): The number of threads to use for processing the images.

    Returns:
        Dict[str, str]: A dictionary where the keys are formatted as "{dataset_name}_IMG_{key}" and the values are base64 encoded string representations of the processed images.
    """
    futures = {}
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        process_bar = tqdm(total=len(images), desc="Processing images", unit="image")
        for key, img in images.items():
            new_key = get_image_id(key, dataset_name)
            futures[new_key] = executor.submit(process_image, img)
        results = {}
        for key, future in futures.items():
            results[key] = future.result()
            process_bar.update(1)
        process_bar.close()
        return results


def frame_video(video_file: str, fps=1):
    """
    Extracts frames from a video file at a specified frame rate and returns them as base64 encoded strings.

    Args:
        video_file (str): The path to the video file.
        fps (int): The frame rate at which frames should be extracted. Defaults to 1 frame per second.

    Returns:
        List[Image]: A list of PIL.Image.Image objects representing the extracted frames.
    """
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
            frames.append(Image.open(BytesIO(buffer)))
            saved_frame_count += 1

        frame_count += 1

    cap.release()

    return frames


def get_image_name(image_path: str) -> str:
    """
    Extracts the image name from a given image path.

    Args:
        image_path (str): The path to the image.

    Returns:
        str: The image name.
    """
    return image_path.split("/")[-1].split(".")[0]


def create_folder(folder_name: str):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
