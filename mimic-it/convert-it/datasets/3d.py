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
            image_path (str): The directory path of the folder named "scannet_frames_25k" obtained by downloading a compressed file from http://www.scan-net.org/ and extracting it.
            num_threads (int): The number of threads to use for processing the images.
        """
        super().__init__(name, short_name, image_path, num_threads)

    def _load_images(self, image_path: str, num_thread: int) -> dict[str, bytes]:
        """
        Loads the images from the dataset.

        Args:
            image_path (str): The path to the directory containing the images downloaded from http://www.scan-net.org/.
            num_threads (int): The number of threads to use for processing the images.

        Returns:
            dict[str, bytes]: A dictionary where the keys are image identifiers and the values are byte strings representing the images.
        """
        from datasets.utils.scene_navigation_utils import process_data

        return process_data(image_path, num_thread)
