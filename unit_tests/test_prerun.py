# unit_tests/test_preload.py
import pytest
import os
import yaml

# Define a custom pytest marker
pytestmark = pytest.mark.prerun


def validate_dataset_structure(yaml_data):
    """
    Validate the dataset info structure
    """
    dataset_info
    expected_keys = {"IMAGE_TEXT", "TEXT_ONLY", "VIDEO_TEXT", "IMAGE_TEXT_IN_CONTEXT"}
    assert set(dataset_info.keys()) == expected_keys, f"Expected keys {expected_keys}, but got {set(dataset_info.keys())}"

    for key, value in dataset_info.items():
        for dataset_name, data in value.items():
            # Ensure all paths with '_path' suffix exist
            for path_key, path_value in data.items():
                assert os.path.exists(path_value), f"Dataset path {path_value} specified under {key} -> {dataset_name} does not exist."


@pytest.fixture
def yaml_data(request):
    yaml_path = request.config.getoption("--yaml-path")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def test_preload_dataset(yaml_data):
    dataset_info = preload_dataset(yaml_data)
    validate_dataset_structure(dataset_info)


# Add CLI option for pytest
def pytest_addoption(parser):
    parser.addoption("--yaml-path", action="store", default="", help="Path to the training data YAML.")
