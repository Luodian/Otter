import base64
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Generator, Tuple

import cv2
from PIL import Image
from tqdm import tqdm


def get_image_id(image_name: str, dataset_name: str) -> str:
    """
    Extracts the image identifier from a given image name.

    Args:
        image_name (str): The name of the image.
        dataset_name (str): The name of the dataset.

    Returns:
        str: The image identifier.
    """
    return f"{dataset_name}_IMG_{get_image_name(image_name)}"


def image_to_bytes(image: Image.Image) -> bytes:
    image_stream = BytesIO()
    image.save(image_stream, format="PNG")
    image_bytes = image_stream.getvalue()
    image_stream.close()
    return image_bytes


def resize_image(img: bytes, target_size: tuple[int, int] = (224, 224)) -> bytes:
    with Image.open(BytesIO(img)) as image:
        if image.size != target_size:
            resized_image = image.resize(target_size, Image.LANCZOS)
            image.close()
            image = resized_image
        resized_image_bytes = image_to_bytes(image)
    return resized_image_bytes


def process_image(image: bytes, target_size=(224, 224)) -> bytes:
    """
    Processes the input image by resizing it, converting it to RGB mode, and save as byte string.

    Args:
        image (bytes): The input image to be processed.

    Returns:
        bytes: The processed image as a byte string.
    """
    with Image.open(BytesIO(image)) as img:
        if img.size != target_size:
            resized_img = img.resize(target_size, Image.LANCZOS)
            img.close()
            img = resized_img
        if img.mode != "RGB":
            converted_img = img.convert("RGB")
            img.close()
            img = converted_img
        processed_image = image_to_bytes(img)
    return processed_image


def get_b64_data(image: bytes) -> str:
    """
    Converts an image to a base64 encoded string.

    Args:
        image (bytes): the image to be converted.

    Returns:
        str: the base64 encoded string representation of the image.
    """
    return base64.b64encode(image).decode("utf-8")


def get_json_data_generator(images: dict[str, bytes], dataset_name: str, num_threads: int) -> Generator[Tuple[str, str], None, None]:
    """
    Converts a dictionary of images to a JSON-compatible dictionary with base64 encoded strings.
    This generator function will yield the processed image data one at a time, allowing you to write the results to a file without needing to store the entire dictionary in memory.
    Args:
        images (Dict[str, bytes]): A dictionary of images, where the keys are image identifiers and the values are byte strings.
        dataset_name (str): The name of the dataset.
        num_threads (int): The number of threads to use for processing the images.

    Returns:
        Dict[str, str]: A dictionary where the keys are formatted as "{dataset_name}_IMG_{key}" and the values are base64 encoded string representations of the processed images.
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        process_bar = tqdm(total=len(images), desc="Processing images", unit="image")

        def process_image_wrapper(args):
            key, img = args
            new_key = get_image_id(key, dataset_name)
            result = get_b64_data(process_image(img))

            process_bar.update()
            return new_key, result

        for result in executor.map(process_image_wrapper, images.items()):
            yield result

        process_bar.close()


def frame_video(video_file: str, fps: int = 1) -> list[bytes]:
    """
    Extracts frames from a video file at a specified frame rate and returns them as base64 encoded strings.

    Args:
        video_file (str): The path to the video file.
        fps (int): The frame rate at which frames should be extracted. Defaults to 1 frame per second.

    Returns:
        List[bytes]: A list of byte strings representing the extracted frames.
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
            # Check if the frame resolution is not 224x224 and resize if necessary
            if frame.shape[0] != 224 or frame.shape[1] != 224:
                frame = cv2.resize(frame, (224, 224))

            success, buffer = cv2.imencode(".png", frame)
            if not success:
                print(f"Failed to encode frame {frame_count} of video {video_file}.")
            frames.append(process_image(buffer))
            saved_frame_count += 1

            del buffer

        frame_count += 1

        del frame

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
    """
    Creates a folder if it does not already exist.

    Args:
        folder_name (str): The name of the folder to create.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
