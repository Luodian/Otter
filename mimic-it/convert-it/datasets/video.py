import json
import os
import gc

from abstract_dataset import AbstractDataset
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from image_utils import frame_video, get_image_name, resize_image, image_to_bytes
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

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, bytes]:
        """
        Loads the images from the dataset.

        Args:
            image_path (str): The path to the directory containing the dataset images.
            num_threads (int): The number of threads to use for processing the images.

        Returns:
            dict[str, bytes]: A dictionary where the keys are image identifiers and the values are image bytes.
        """
        videos = glob(f"{image_path}/*.mp4")
        if len(videos) <= 100:
            raise ValueError("Not enough videos in the dataset, please check the path.")
        with ThreadPoolExecutor(max_workers=num_thread) as executor:
            results = {}
            process_bar = tqdm(total=len(videos), desc="Processing videos into images", unit="video")
            cnt = 0
            for video, framed_results in executor.map(lambda x: (get_image_name(x), frame_video(x)), videos):
                for index, result in enumerate(framed_results):
                    # print("video", video)
                    name = video + "_" + str(index).zfill(4)
                    results[name] = result
                process_bar.update(1)

                cnt = cnt + 1
                if cnt % 100 == 0:
                    gc.collect()
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
            name (str): The name of the dataset. Defaults to "VisualStoryTelling".
            short_name (str): The short name of the dataset. Defaults to "VST".
            image_path (str): The path to the JSON file containing the dataset images.
            num_threads (int): The number of threads to use for processing the images.
        """
        super().__init__(name, short_name, image_path, num_threads)

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, bytes]:
        """
        Loads the images from the VisualStoryTelling dataset.

        Args:
            image_path (str): The path to the JSON file containing the dataset images.
            num_threads (int): The number of threads to use for processing the images.

        Returns:
            dict[str, bytes]: A dictionary of images, where the keys are the IDs of the images.
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

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, bytes]:
        """
        Load the images from the TVCaptions dataset.

        Args:
            image_path (str): The path to the directory containing the dataset images, downloaded from https://tvqa.cs.unc.edu/download_tvqa.html#tvqa-download-4.
            num_threads (int): The number of threads to use for loading the images.

        Returns:
            Dict[str, Image.Image]: A dictionary of images, where the keys are the IDs of the images.
        """

        def get_frames(directory, frames=16):
            """
            Generate a list of image filenames.

            Args:
                directory (str): The directory path containing the frames.
                frames (int): The number of frames to sample. Defaults to 16.

            Returns:
                list[str]: A list of image filenames.
            """

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
            """
            Load and resize images from frames.

            Args:
                frames (list[str]): List of image filenames.
                frame_name (str): Name of the frame.
                clip_name (str): Name of the clip.

            Returns:
                dict[str, bytes]: A dictionary where the keys are image identifiers and the values are image bytes.
            """

            images = {}
            for frame in frames:
                image_name = os.path.basename(frame).split(".")[0]
                # print(frame_name, clip_name, image_name)
                if clip_name.startswith(frame_name):
                    image_id = f"{clip_name}_{image_name}"
                else:
                    image_id = f"{frame_name}_{clip_name}_{image_name}"
                # print(image_id)
                with open(frame, "rb") as f:
                    frame_bytes = f.read()
                images[image_id] = resize_image(frame_bytes)
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
                    progress_bar.update()
                    # print("The type of all_images is", type(all_images))
                    # print("The types of the keys and values of all_images are", type(list(all_images.keys())[0]), type(list(all_images.values())[0]))
            progress_bar.close()

        return all_images
