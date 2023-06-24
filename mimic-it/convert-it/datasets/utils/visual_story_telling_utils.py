import requests

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
from image_utils import resize_image


def get_url(image: dict[str]):
    if "url_o" in image:
        return image["url_o"]
    else:
        return image["url_m"]


def download_single_image(image: dict[str]) -> tuple[str, bytes]:
    url = get_url(image)
    id = image["id"]
    pic = requests.get(url)
    with Image.open(BytesIO(pic.content)) as img:
        return id, resize_image(img).tobytes()


def download(images: list[dict[str]], num_threads: int):
    output = {}
    process_bar = tqdm(total=len(images), unit="image", desc="Downloading images")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for id, image in executor.map(download_single_image, images):
            output[id] = image
            process_bar.update(1)
    process_bar.close()
    return output
