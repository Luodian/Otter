import base64
import os
import cv2

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


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


def resize_image(image: Image.Image, target_size: tuple[int, int] = (224, 224)) -> Image.Image:
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


def process_image(image: bytes) -> str:
    """
    Processes the input image by resizing it, converting it to RGB mode, and encoding it as base64.

    Args:
        image (bytes): The input image to be processed.

    Returns:
        str: The base64 encoded string representation of the processed image.
    """
    with Image.open(BytesIO(image)) as img:
        resized_img = resize_image(img)
    if resized_img.mode != "RGB":
        resized_img = resized_img.convert("RGB")
    buffer = BytesIO()
    resized_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64


def get_json_data(images: dict[str, bytes], dataset_name: str, num_threads: int) -> dict[str, str]:
    """
    Converts a dictionary of images to a JSON-compatible dictionary with base64 encoded strings.

    Args:
        images (Dict[str, bytes]): A dictionary of images, where the keys are image identifiers and the values are byte strings.
        dataset_name (str): The name of the dataset.
        num_threads (int): The number of threads to use for processing the images.

    Returns:
        Dict[str, str]: A dictionary where the keys are formatted as "{dataset_name}_IMG_{key}" and the values are base64 encoded string representations of the processed images.
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        process_bar = tqdm(total=len(images), desc="Processing images", unit="image")
        results = {}

        def process_image_wrapper(args):
            key, img = args
            new_key = get_image_id(key, dataset_name)
            result = process_image(img)

            process_bar.update(1)
            return new_key, result

        processed_images = executor.map(process_image_wrapper, images.items())

        for key, result in processed_images:
            results[key] = result

        process_bar.close()

        return results


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
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                print(f"Failed to encode frame {frame_count} of video {video_file}.")
            frames.append(process_image(buffer))
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
    """
    Creates a folder if it does not already exist.

    Args:
        folder_name (str): The name of the folder to create.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
