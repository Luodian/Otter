import os

from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from image_utils import resize_image


def process(cur_dir, img_root):
    root = os.path.join(img_root, cur_dir, "color")
    file_list = os.listdir(root)
    images = {}
    for cur_file in file_list:
        file_name = os.path.join(img_root, cur_dir, "color", cur_file)
        img = Image.open(file_name) # path to file
        image_id = f"{cur_dir}_color_{cur_file[:-4]}"
        images[image_id] = resize_image(img)
    return images


def process_data(img_root: str, num_threads: int):
    keys = os.listdir(img_root)
    all_images = {}
    process_bar = tqdm(total=len(keys), unit="image", desc="Loading images")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process, cur_dir, img_root) for cur_dir in keys]
        for future in futures:
            images = future.result()
            all_images.update(images)
            process_bar.update(1)
    process_bar.close()
    return all_images
