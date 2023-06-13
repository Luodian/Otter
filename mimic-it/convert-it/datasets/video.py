import json

from abstract_dataset import AbstractDataset
from PIL import Image
from typing import Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from image_utils import (
    frame_video,
    get_image_name,
    get_image_id
)


class DenseCaptions(AbstractDataset):
    def __init__(
        self,
        name: str = "DenseCaptions",
        *,
        image_path: str
    ):
        super().__init__(name, image_path)

    def _load_images(
        self, image_path: str, num_thread: int
    ) -> Dict[str, Image]:
        videos = glob(f"{image_path}/*.mp4")
        if len(videos) <= 100:
            raise ValueError("Not enough videos in the dataset, please check the path.")
        with ThreadPoolExecutor() as executor:
            futures = {}
            for video in videos:
                futures[get_image_name] = executor.submit(frame_video, video)
            results = {}
            process_bar = tqdm(total=len(videos), desc="Processing videos into images", unit="video")
            for video, future in futures.items():
                for index, result in enumerate(future.result()):
                    name = get_image_id(
                        self.name,
                        video + "_" + str(index).zfill(4)
                    )
                    results[name] = result
                process_bar.update(1)
            process_bar.close()
            return results