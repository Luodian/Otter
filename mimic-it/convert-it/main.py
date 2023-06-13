import os
import argparse
import json

from abstract_dataset import get_dataset_by_path
from image_utils import get_json_data, create_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, required=True, help="Path to the dataset class."
    )
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads.")
    parser.add_argument("--image_path", help="Path to the prompt file.")

    args = parser.parse_args()
    dataset_args = {}
    if args.image_path is not None:
        dataset_args["image_path"] = args.image_path
    if args.num_threads is not None:
        dataset_args["num_threads"] = args.num_threads
    dataset = get_dataset_by_path(args.name, dataset_args)
    dataset_short_name = dataset.short_name
    dataset = dict(dataset)
    json_data = get_json_data(dataset, dataset_short_name, args.num_threads)
    create_folder("output")
    with open(f"output/{dataset_short_name}.json", "w") as f:
        json.dump(json_data, f)
