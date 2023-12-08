import json
import requests

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from image_utils import resize_image, create_folder


def get_url(image: dict[str]):
    """
    Retrieve the URL of the image.

    Args:
        image: A dictionary containing image information.

    Returns:
        The URL of the image.

    """
    if "url_o" in image:
        return image["url_o"]
    else:
        return image["url_m"]


def download_single_image(image: dict[str]) -> tuple[str, bytes]:
    """
    Download a single image and resize it.

    Args:
        image: A dictionary containing image information.

    Returns:
        A tuple containing the image ID and the resized image as bytes.

    """
    url = get_url(image)
    id = image["id"]
    try:
        pic = requests.get(url)
        return id, resize_image(pic.content)
    except:
        return id, None


def download(images: list[dict[str]], num_threads: int):
    """
    Download multiple images concurrently using thread pooling.

    Args:
        images: A list of dictionaries, each containing image information.
        num_threads: The number of threads to use for concurrent downloading.

    Returns:
        A dictionary mapping image IDs to their corresponding resized images as bytes.

    """
    output = {}
    process_bar = tqdm(total=len(images), unit="image", desc="Downloading images")
    expired_images = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for id, image in executor.map(download_single_image, images):
            if image is not None:
                output[id] = image
            else:
                expired_images.append(id)
            process_bar.update(1)
    process_bar.close()
    create_folder("output")
    with open("output/expired_images.json", "w") as f:
        json.dump(expired_images, f, indent=4)
    return output
