from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from PIL.Image import Image
import importlib

AVAILABLE_DATASETS: List[str] = [
    "change.SpotTheDifference",
    "video.DenseCaptions",
    "video.TVCaptions",
    "video.VisualStoryTelling",
    "3d.SceneNavigation",
    "fpv.EGO4D",
]


class AbstractDataset(ABC):
    def __init__(self, name: str, short_name: str, image_path: str, num_threads: int):
        """
        Constructor.

        Args:
            name (str): The name of the dataset.
            image_path (str): The path to the images of the dataset.
        """
        self.name: str = name
        self.short_name: str = short_name
        self.images: Dict[str, Image] = self._load_images(image_path, num_threads)

    @abstractmethod
    def _load_images(self, image_path: str, num_thread: int) -> dict[str, Image]:
        """
        Load the images from the videos or albums.

        Args:
            image_path (str): The path storing the videos or albums.
            num_thread (int): The number of threads to use for loading the images.

        Returns:
            Dict[str, Image]: A dictionary of images, where the keys are the IDs of the images.
        """
        pass

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """
        Return the item at the given index as a dictionary.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Dict[str, Any]: The item at the given index.
        """
        return self.images[key]

    def __iter__(self) -> "AbstractDataset":
        """
        Return the iterator object for the dataset.

        Returns:
            AbstractDataset: The iterator object.
        """
        self.keys = iter(self.images.keys())
        return self

    def __next__(self) -> Tuple[str, Image]:
        """
        Return the next item in the iteration.

        Returns:
            Tuple[str, Image]: The next item as a tuple of index and image.

        Raises:
            StopIteration: If there are no more items in the iteration.
        """
        try:
            key = next(self.keys)
            image = self.images[key]
            return key, image
        except StopIteration:
            raise StopIteration

    def __len__(self) -> int:
        return len(self.query_inputs)

    def __str__(self) -> str:
        return f"{self.name} dataset"


def get_dataset_by_path(path: str, dataset_args: dict[str, str]) -> AbstractDataset:
    assert path in AVAILABLE_DATASETS, f"{path} is not an available dataset."
    module_path, dataset_name = path.split(".")
    module_path = "datasets." + module_path

    # Import the module and load the class

    imported_module = importlib.import_module(module_path)
    dataset_class = getattr(imported_module, dataset_name)

    # Instantiate the class and return the instance
    return dataset_class(**dataset_args)


def get_available_datasets() -> List[str]:
    return AVAILABLE_DATASETS
