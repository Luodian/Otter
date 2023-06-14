from glob import glob
from PIL import Image

from abstract_dataset import AbstractDataset


class SceneNavigation(AbstractDataset):
    def __init__(
        self,
        name: str = "SceneNavigation",
        short_name="SN",
        *,
        image_path: str,
        num_threads: int,
    ):
        """
        Initializes a SceneNavigation dataset.

        Args:
            name (str): The name of the dataset. Defaults to "SceneNavigation".
            short_name (str): The short name of the dataset. Defaults to "SN".
            image_path (str): The path to the directory containing the dataset images.
            num_threads (int): The number of threads to use for processing the images.
        """
        super().__init__(name, short_name, image_path, num_threads)

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, Image.Image]:
        """
        Loads the images from the dataset.

        Args:
            image_path (str): The path to the directory containing the images downloaded from http://www.scan-net.org/.
            num_threads (int): The number of threads to use for processing the images.

        Returns:
            dict[str, Image.Image]: A dictionary where the keys are image identifiers and the values are PIL.Image.Image objects.
        """
        from datasets.utils.scene_navigation_utils import process_data
        
        return process_data(image_path, num_thread)
