import json

from abstract_dataset import AbstractDataset
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from image_utils import frame_video, get_image_name


class DenseCaptions(AbstractDataset):
    def __init__(
        self,
        name: str = "DenseCaptions",
        short_name="DC",
        *,
        image_path: str,
        num_threads: int,
    ):
        """
        Initializes a DenseCaptions dataset.

        Args:
            name (str): The name of the dataset. Defaults to "DenseCaptions".
            short_name (str): The short name of the dataset. Defaults to "DC".
            image_path (str): The path to the directory containing the dataset images.
            num_threads (int): The number of threads to use for processing the images.
        """
        super().__init__(name, short_name, image_path, num_threads)

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, Image.Image]:
        """
        Loads the images from the dataset.

        Args:
            image_path (str): The path to the directory containing the all the videos downloaded from ActivityNet (in mp4 format).
            num_threads (int): The number of threads to use for processing the images.

        Returns:
            dict[str, Image.Image]: A dictionary where the keys are image identifiers and the values are PIL.Image.Image objects.
        """
        videos = glob(f"{image_path}/*.mp4")
        if len(videos) <= 100:
            raise ValueError("Not enough videos in the dataset, please check the path.")
        with ThreadPoolExecutor(max_workers=num_thread) as executor:
            futures = {}
            for video in videos:
                futures[get_image_name(video)] = executor.submit(frame_video, video)
            results = {}
            process_bar = tqdm(
                total=len(videos), desc="Processing videos into images", unit="video"
            )
            for video, future in futures.items():
                for index, result in enumerate(future.result()):
                    # print("video", video)
                    name = video + "_" + str(index).zfill(4)
                    results[name] = result
                process_bar.update(1)
            process_bar.close()
            return results


class VisualStoryTelling(AbstractDataset):
    def __init__(
        self,
        name: str = "VisualStroryTelling",
        short_name="VST",
        *,
        image_path: str,
        num_threads: int,
    ):
        """
        Initializes a DenseCaptions dataset.

        Args:
            name (str): The name of the dataset. Defaults to "VisualStroryTelling".
            short_name (str): The short name of the dataset. Defaults to "VST".
            image_path (str): The json file (train.story-in-sequence.json) containing the dataset images, downloaded from https://visionandlanguage.net/VIST.
            num_threads (int): The number of threads to use for processing the images.
        """
        super().__init__(name, short_name, image_path, num_threads)
    
    def _load_images(self, image_path: str, num_thread: int) -> dict[str, Image.Image]:
        from datasets.visual_story_telling_utils import download

        with open(image_path, "r") as f:
            data = json.load(f)
        return download(data["images"], num_thread)
