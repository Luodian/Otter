import json
import os

from abstract_dataset import AbstractDataset
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from image_utils import frame_video, get_image_name, resize_image
from natsort import natsorted


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
            results = {}
            process_bar = tqdm(total=len(videos), desc="Processing videos into images", unit="video")
            for video, framed_results in executor.map(lambda x: (get_image_name(x), frame_video(x)), videos):
                for index, result in enumerate(framed_results):
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
        Initializes a VisualStoryTelling dataset.

        Args:
            name (str): The name of the dataset. Defaults to "VisualStroryTelling".
            short_name (str): The short name of the dataset. Defaults to "VST".
            image_path (str): The json file (train.story-in-sequence.json) containing the dataset images, downloaded from https://visionandlanguage.net/VIST.
            num_threads (int): The number of threads to use for processing the images.
        """
        super().__init__(name, short_name, image_path, num_threads)

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, Image.Image]:
        """
        Load the images from the VisualStoryTelling dataset.

        Args:
            image_path (str): The path to the JSON file containing the dataset images.
            num_thread (int): The number of threads to use for loading the images.

        Returns:
            Dict[str, Image.Image]: A dictionary of images, where the keys are the IDs of the images.
        """
        from datasets.utils.visual_story_telling_utils import download

        with open(image_path, "r") as f:
            data = json.load(f)
        return download(data["images"], num_thread)


class TVCaptions(AbstractDataset):
    def __init__(
        self,
        name: str = "TVCaptions",
        short_name="TVC",
        *,
        image_path: str,
        num_threads: int,
    ):
        """
        Initializes a TVCaptions dataset.

        Args:
            name (str): The name of the dataset. Defaults to "TVCaptions".
            short_name (str): The short name of the dataset. Defaults to "TVC".
            image_path (str): The path to the directory containing the dataset images, downloaded from https://tvqa.cs.unc.edu/download_tvqa.html#tvqa-download-4
            num_threads (int): The number of threads to use for processing the images.
        """
        super().__init__(name, short_name, image_path, num_threads)

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, Image.Image]:
        """
        Load the images from the TVCaptions dataset.

        Args:
            image_path (str): The path to the directory containing the dataset images, downloaded from https://tvqa.cs.unc.edu/download_tvqa.html#tvqa-download-4.
            num_thread (int): The number of threads to use for loading the images.

        Returns:
            Dict[str, Image.Image]: A dictionary of images, where the keys are the IDs of the images.

        """

        def get_frames(directory, frames=16):
            # Generate a list of image filenames
            image_filenames = natsorted(glob(os.path.join(directory, "*")))

            # Calculate the stride length to achieve an average sample
            stride = max(1, len(image_filenames) // frames)

            # Initialize the starting index for sampling
            start_index = stride // 2

            # Sample 16 images evenly
            sampled_images = [image_filenames[i] for i in range(start_index, len(image_filenames), stride)]

            return sampled_images

        def get_images(frames, frame_name, clip_name):
            images = {}
            for frame in frames:
                image_name = os.path.basename(frame).split(".")[0]
                if clip_name.startswith(frame_name):
                    image_id = f"{clip_name}_{image_name}"
                else:
                    image_id = f"{frame_name}_{clip_name}_{image_name}"
                images[image_id] = resize_image(Image.open(frame))
            return images

        frames = glob(os.path.join(image_path, "*"))
        all_images = {}
        for frame in frames:
            frame_name = os.path.basename(frame).split("_")[0]
            clips = glob(os.path.join(frame, "*"))
            progress_bar = tqdm(total=len(clips), desc=f"Processing clips in {frame_name}", unit="clip")
            with ThreadPoolExecutor(max_workers=num_thread) as executor:

                def get_images_dict(clip):
                    clip_name = os.path.basename(clip)
                    frames = get_frames(clip)
                    return get_images(frames, frame_name, clip_name)

                for images in executor.map(get_images_dict, clips):
                    all_images.update(images)
                    progress_bar.update(1)
            progress_bar.close()

        return all_images
