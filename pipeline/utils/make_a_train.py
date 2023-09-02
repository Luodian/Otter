import json
import random
import orjson
import argparse
from tqdm import tqdm


def main(input_file, output_file):
    # Load the JSON file
    with open(input_file, "rb") as file:
        data = orjson.loads(file.read())

    # Create a set to store seen keys
    seen_keys = set()

    # Create a new dictionary with the keys from the original JSON and rel_ins_ids as values
    new_dict = {}
    for key, value in tqdm(data["data"].items()):
        if key not in seen_keys:
            try:
                # Check if rel_ins_ids are in the original JSON
                valid_rel_ins_ids = [rel_ins_id for rel_ins_id in value["rel_ins_ids"] if rel_ins_id in data["data"]]

                # Add the valid rel_ins_ids to the new_dict
                new_dict[key] = valid_rel_ins_ids
                seen_keys.update(valid_rel_ins_ids)
            except Exception as e:
                print("Error with key %s and value %s" % (key, value))

    # Write the new dictionary to a new JSON file
    with open(output_file, "wb") as file:
        file.write(orjson.dumps(new_dict))


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process a JSON file.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str, help="Path to the output JSON file")

    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.input_file, args.output_file)
