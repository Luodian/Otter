from abstract_dataset import AbstractDataset
from PIL.Image import Image
from typing import Dict
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
        super().__init__(name, short_name, image_path, num_threads)

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, Image]:
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
