import sys
import argparse
import os
import yaml
import contextlib

sys.path.append("../..")
from .models.base_model import load_model
from .datasets.base_eval_dataset import load_dataset


def get_info(info):
    if "name" not in info:
        raise ValueError("Model name is not specified.")
    name = info["name"]
    # info.pop("name")
    return name, info


def load_models(model_infos):
    for model_info in model_infos:
        name, info = get_info(model_info)
        model = load_model(name, info)
        yield model


def load_datasets(dataset_infos):
    for dataset_info in dataset_infos:
        name, info = get_info(dataset_info)
        dataset = load_dataset(name, info)
        yield dataset


class DualOutput:
    def __init__(self, file, stdout):
        self.file = file
        self.stdout = stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to the config file, suppors more specific configurations.",
        default=None,
    )
    args.add_argument(
        "--models",
        type=str,
        nargs="?",
        help="Specify model names as comma separated values.",
        default=None,
    )
    args.add_argument(
        "--model_paths",
        type=str,
        nargs="?",
        help="Specify model paths as comma separated values.",
        default=None,
    )
    args.add_argument(
        "--datasets",
        type=str,
        nargs="?",
        help="Specify dataset names as comma separated values.",
        default=None,
    )
    args.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for logging results.",
        default="./logs/evaluation.txt",
    )
    args.add_argument(
        "--cache_dir",
        type=str,
        help="Cache directory for datasets.",
        default=None,
    )

    phrased_args = args.parse_args()

    if phrased_args.config:
        with open(phrased_args.config, "r") as f:
            config = yaml.safe_load(f)
        model_infos = config["models"]
        dataset_infos = config["datasets"]
        phrased_args.output = config["output"] if "output" in config else phrased_args.output
    else:
        # Zip the models and their respective paths
        model_names = phrased_args.models.split(",")
        if phrased_args.model_paths is not None:
            model_paths = phrased_args.model_paths.split(",")
            model_infos = [{"name": name, "model_path": path} for name, path in zip(model_names, model_paths)]
        else:
            model_infos = [{"name": name} for name in model_names]
        dataset_infos = [{"name": dataset_name, "cache_dir": phrased_args.cache_dir} for dataset_name in phrased_args.datasets.split(",")]

    if not os.path.exists(os.path.dirname(phrased_args.output)):
        os.makedirs(os.path.dirname(phrased_args.output))

    with open(phrased_args.output, "w") as outfile, contextlib.redirect_stdout(DualOutput(outfile, sys.stdout)):
        print("=" * 80)
        print(" " * 30 + "EVALUATION REPORT")
        print("=" * 80)
        print()

        for model_info in model_infos:
            print("\nMODEL INFO:", model_info)
            print("-" * 80)
            model = load_model(model_info["name"], model_info)

            dataset_count = 0
            for dataset in load_datasets(dataset_infos):
                dataset_count += 1
                print(f"\nDATASET: {dataset.name}")
                print("-" * 20)

                dataset.evaluate(model)  # Assuming this function now prints results directly.
                print()

            print("-" * 80)
            print(f"Total Datasets Evaluated: {dataset_count}\n")

        print("=" * 80)

# python evaluate.py --models otter_image --datasets mmbench
