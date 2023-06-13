import os
import json

from abstract_dataset import AbstractDataset
from PIL import Image
from tqdm import tqdm
from glob import glob
from image_utils import create_folder


class SpotTheDifference(AbstractDataset):
    def __init__(
        self,
        name: str = "SpotTheDifference",
        short_name="SD",
        *,
        image_path: str,
        num_threads: int,
    ):
        """
        Initializes a SpotTheDifference dataset.

        Args:
            name (str): The name of the dataset. Defaults to "SpotTheDifference".
            short_name (str): The short name of the dataset. Defaults to "SD".
            image_path (str): The path containing the dataset images, downloaded from https://drive.google.com/file/d/1OVb4_3Uec_xbyUk90aWC6LFpKsIOtR7v/view?usp=sharing.
            num_threads (int): The number of threads to use for processing the images.
        """
        super().__init__(name, short_name, image_path, num_threads)

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, Image.Image]:
        """
        Loads the images from the dataset.

        Args:
            image_path (str): The path to the dictionary containing the dataset images.
            num_threads (int): The number of threads to use for processing the images.

        Returns:
            dict[str, Image.Image]: A dictionary where the keys are image identifiers and the values are PIL.Image.Image objects.
        """
        file_names = glob(os.path.join(image_path, "*"))
        names = set()
        for file_name in file_names:
            image_name = file_name.split("/")[-1].split(".")[0]
            id = image_name.split("_")[0]
            names.add(id)
        ids = list(sorted(list(names)))[:5]

        jpgs_path = glob(os.path.join(image_path, "*.jpg"))
        jpegs_path = glob(os.path.join(image_path, "*.jpeg"))
        pngs_path = glob(os.path.join(image_path, "*.png"))
        jpgs = set()
        pngs = set()
        for path in jpgs_path:
            jpgs.add(path.split("/")[-1].split(".")[0])
        for path in pngs_path:
            pngs.add(path.split("/")[-1].split(".")[0])
            
        def get_path(file_name):
            if file_name in jpgs:
                return os.path.join(image_path, file_name + ".jpg")
            elif file_name in pngs:
                # print("file_name", file_name, os.path.join(image_path, file_name + ".png"))
                return os.path.join(image_path, file_name + ".png")
            elif file_name in jpegs_path:
                return os.path.join(image_path, file_name + ".jpeg")
            else:
                # print("===================================", file_name)
                raise Exception("File not found")
        
        def read_image(file_name) -> Image.Image:
            return Image.open(file_name)
        
        file_not_found = []
        
        images = {}
        
        for id in tqdm(ids, desc="Reading images"):
            try:
                file_1 = get_path(id)
                file_2 = get_path(id + "_2")
                # print(file_1, file_2)
                images[id.zfill(5) + "_1"] = read_image(file_1)
                images[id.zfill(5) + "_2"] = read_image(file_2)
            except Exception as e:
                file_not_found.append(id)
                print(f"File not found: {id}")
                # print(f"Error: {e}")

        create_folder("log")
        with open("log/file_not_found.log", "w") as f:
            json.dump(file_not_found, f, indent=4)

        return images
