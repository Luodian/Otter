import os

from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from image_utils import process_image


def process(cur_dir, img_root):
    """
    Process images in a directory.

    Args:
        cur_dir (str): The name of the current directory.
        img_root (str): The root directory of the images.

    Returns:
        dict: A dictionary containing processed images. The keys are unique identifiers
        for each image, and the values are the processed images.

    """
    root = os.path.join(img_root, cur_dir, "color")
    file_list = os.listdir(root)
    images = {}
    for cur_file in file_list:
        file_name = os.path.join(img_root, cur_dir, "color", cur_file)
        with open(file_name, "rb") as f:
            img = f.read()
        image_id = f"{cur_dir}_color_{cur_file[:-4]}"
        images[image_id] = process_image(img)
    return images


def process_data(img_root: str, num_threads: int):
    """
    Process images in parallel using multiple threads.

    Args:
        img_root (str): The root directory of the images.
        num_threads (int): The number of threads to use for parallel processing.

    Returns:
        dict: A dictionary containing processed images. The keys are unique identifiers
        for each image, and the values are the processed images.

    """
    keys_dir = glob(os.path.join(img_root, "scene*_00"))
    keys = list(map(os.path.basename, keys_dir))
    all_images = {}
    process_bar = tqdm(total=len(keys), unit="image", desc="Loading images")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for images in executor.map(process, keys, [img_root] * len(keys)):
            all_images.update(images)
            process_bar.update()
    process_bar.close()
    return all_images
