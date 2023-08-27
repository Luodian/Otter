import json

from abstract_dataset import AbstractDataset


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
            image_root (str): The root path to the COCO image train split.
            image_path (str): The path to the JSON file containing the dataset images.
                              The images can be downloaded from:
                              https://drive.google.com/file/d/1OVb4_3Uec_xbyUk90aWC6LFpKsIOtR7v/view?usp=sharing.
            num_threads (int): The number of threads to use for processing the images.
        """
        self.image_root = image_root
        super().__init__(name, short_name, image_path, num_threads)

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, bytes]:
        """
        Loads the images from the dataset.

        Args:
            image_path (str): The path to the JSON file containing the dataset images.
            num_threads (int): The number of threads to use for processing the images.

        Returns:
            dict[str, bytes]: A dictionary where the keys are image identifiers and the values are bytes objects representing the images.
        """

        def read_image(file_name) -> bytes:
            with open(file_name, "rb") as f:
                return f.read()

        images = {}
        with open(image_path) as f:
            image_ids = json.load(f).keys()

        for cur_image_id in image_ids:
            images[cur_image_id] = read_image(f"{self.image_root}/{cur_image_id}.jpg")

        return images
