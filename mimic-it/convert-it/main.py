import argparse
import orjson

from abstract_dataset import get_dataset_by_path
from image_utils import get_json_data_generator, create_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Path to the dataset class.")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads.")
    parser.add_argument("--image_path", help="Path to the prompt file.")
    parser.add_argument("--image_root", default=None, help="Path to the image root.")

    args = parser.parse_args()
    dataset_args = {}
    if args.image_path is not None:
        dataset_args["image_path"] = args.image_path
    if args.num_threads is not None:
        dataset_args["num_threads"] = args.num_threads
    if args.image_root is not None:
        dataset_args["image_root"] = args.image_root
    dataset = get_dataset_by_path(args.name, dataset_args)
    dataset_short_name = dataset.short_name
    dataset = dict(dataset)
    create_folder("output")

    # Open the output JSON file in text mode, since we'll be writing strings
    with open(f"output/{dataset_short_name}.json", "w") as f:
        # Write the opening brace for the JSON object
        f.write("{")

        # Use a flag to track whether a comma is needed before the next key-value pair
        need_comma = False

        # Iterate over the generator, which yields key-value pairs one at a time
        for image_key, base64_data in get_json_data_generator(dataset, dataset_short_name, args.num_threads):
            # Write a comma before the next key-value pair if needed
            if need_comma:
                f.write(", ")

            # Write the key-value pair as a string to the file
            f.write(f'"{image_key}": "{base64_data}"')

            # Set the flag to True so that a comma is written before the next key-value pair
            need_comma = True

        # Write the closing brace for the JSON object
        f.write("}")
