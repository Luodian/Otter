import unittest
from unittest.mock import Mock
from pipeline.mimicit_utils.mmc4_dataset import get_mmc4_dataset


class TestGetMMC4Dataset(unittest.TestCase):
    def test_get_mmc4_dataset(self):
        # Mock the required inputs
        args = Mock(
            mmc4_shards="/home/luodian/projects/Otter/archived/000000000.tar",
            train_num_samples_mmc4=1000,
            mmc4_textsim_threshold=0.32,
            batch_size_mmc4=10,
            seed=0,
            workers=2,
            world_size=1,
        )
        image_processor = Mock()
        tokenizer = Mock()

        # Call the function to test
        data_info = get_mmc4_dataset(args, image_processor, tokenizer)

        # Check if the dataloader's attributes are as expected
        self.assertEqual(data_info.dataloader.num_batches, 100)
        self.assertEqual(data_info.dataloader.num_samples, 1000)
