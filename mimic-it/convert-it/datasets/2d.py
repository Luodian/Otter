import os
import json

from abstract_dataset import AbstractDataset
from PIL import Image
from tqdm import tqdm
from glob import glob
from image_utils import create_folder


class Llava(AbstractDataset):
    def __init__(
        self,
        name: str = "Llava",
        short_name="LA",
        *,
        image_root: str,
        image_path: str,
        num_threads: int,
    ):
        """
        Initializes a Llava in-context dataset.

        Args:
            name (str): The name of the dataset. Defaults to "Llava".
            short_name (str): The short name of the dataset. Defaults to "LA".
            image_path (str): The path containing the dataset images, downloaded from https://drive.google.com/file/d/1OVb4_3Uec_xbyUk90aWC6LFpKsIOtR7v/view?usp=sharing.
            image_root (str): The path to the coco image train split
            num_threads (int): The number of threads to use for processing the images.
        """
        self.image_root = image_root
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
        def read_image(file_name) -> Image.Image:
            return Image.open(file_name)

        images = {}
        with open(image_path) as f:
            image_ids = json.load(f).keys()

        for cur_image_id in image_ids:
            images[cur_image_id] = read_image(f"{self.image_root}/{cur_image_id}.jpg")

        return images
