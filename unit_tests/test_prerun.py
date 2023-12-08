# Inside tests/unit_tests/test_prerun.py
import os
import yaml
import pytest
import orjson
import pandas as pd


# Define the pytest fixture
@pytest.fixture
def yaml_data(request):
    yaml_path = request.config.getoption("--yaml-path")
    if not yaml_path or not os.path.exists(yaml_path):
        pytest.fail(f"YAML file path '{yaml_path}' does not exist.")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


# Your test function
@pytest.mark.prerun
def test_yaml_structure(yaml_data):
    required_categories = [
        "IMAGE_TEXT",
        "TEXT_ONLY",
        "VIDEO_TEXT",
        "IMAGE_TEXT_IN_CONTEXT",
    ]

    for category, datasets in yaml_data.items():
        assert category in required_categories, f"Unexpected category '{category}' in YAML. Expected categories are {required_categories}."

        for dataset_name, data in datasets.items():
            for path_key, path_value in data.items():
                if path_key.endswith("_path"):
                    assert os.path.exists(path_value), f"Dataset path {path_value} specified under {category} -> {dataset_name} does not exist."
                elif path_key == "num_samples":
                    assert isinstance(path_value, int), f"'num_samples' should be an integer but got {type(path_value)} under {category} -> {dataset_name}."

                # checking mimicit path aligns with corresponding format.
                if path_key == "mimicit_path":
                    print(f"Checking -> {path_value} in MIMICIT format.")
                    with open(path_value, "rb") as f:
                        data = orjson.loads(f.read())

                    assert "data" in data

                if path_key == "images_path":
                    print(f"Checking -> {path_value} in images format.")
                    assert os.path.exists(path_value), f"Dataset path {path_value} specified under {category} -> {dataset_name} does not exist."
                    # # Read the parquet file using pandas
                    # df = pd.read_parquet(path_value)

                    # # Check for the 'base64' column
                    # assert "base64" in df.columns, f"The 'base64' column was not found in the dataset {path_value}."
