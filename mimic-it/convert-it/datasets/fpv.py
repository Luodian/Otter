import os

from glob import glob

from abstract_dataset import AbstractDataset
from image_utils import frame_video

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class EGO4D(AbstractDataset):
    def __init__(
        self,
        name: str = "EGO4D",
        short_name="E4D",
        *,
        image_path: str,
        num_threads: int,
    ):
        """
        Initializes an EGO4D dataset.

        Args:
            name (str): The name of the dataset. Defaults to "EGO4D".
            short_name (str): The short name of the dataset. Defaults to "E4D".
            image_path (str): The directory path of the folder downloaded from https://ego4d-data.org/#download.
            num_threads (int): The number of threads to use for processing the images.
        """
        super().__init__(name, short_name, image_path, num_threads)

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, bytes]:
        """
        Loads the images from the dataset.

        Args:
            image_path (str): The path to the directory containing the images downloaded from https://ego4d-data.org/#download.
            num_threads (int): The number of threads to use for processing the images.

        Returns:
            dict[str, bytes]: A dictionary where the keys are image identifiers and the values are image bytes.

        Raises:
            FileNotFoundError: If the specified image path does not exist.
        """
        video_paths = glob(os.path.join(image_path, "*"))

        def get_image(video_path):
            images = frame_video(video_path)
            images_dict = {}
            video_name = os.path.basename(video_path).split(".")[0]
            for index, image in enumerate(images):
                images_dict[f"{video_name}_{index:08d}"] = image
            return images_dict

        final_images_dict = {}

        with ThreadPoolExecutor(max_workers=num_thread) as executor:
            process_bar = tqdm(total=len(video_paths), unit="video", desc="Processing videos into images")
            for images_dict in executor.map(get_image, video_paths):
                final_images_dict.update(images_dict)
                process_bar.update()
            process_bar.close()

        return final_images_dict
