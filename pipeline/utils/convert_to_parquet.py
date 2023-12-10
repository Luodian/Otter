import pandas as pd
import os
import time
import json
from tqdm import tqdm
import argparse
import orjson
import dask.dataframe as dd
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_images(base64_str, resize_res=-1):
    import base64
    from PIL import Image
    from io import BytesIO

    if not base64_str:
        print("Warning: Empty base64 string encountered.")
        return None

    padding_needed = 4 - len(base64_str) % 4
    if padding_needed != 4:
        base64_str += "=" * padding_needed

    try:
        if resize_res == -1:
            img = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        else:
            img = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB").resize((resize_res, resize_res))
    except Exception as e:
        print(f"Warning: Failed to open image. Error: {e}")
        return None

    if img.mode == "RGBA":
        img = img.convert("RGB")

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    new_base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return new_base64_str


def convert_json_to_parquet(input_path, output_path, max_partition_size):
    start_time = time.time()
    with open(input_path, "rb") as f:
        data = f.read()
        data_dict = orjson.loads(data)

    total_size = len(data)
    print(f"Total size of the JSON data: {total_size} bytes")

    nparitions = int(max(1, total_size // max_partition_size))
    print(f"Number of partitions: {nparitions}")

    resized_data_dict = {}
    dropped_keys = []

    # Initialize the progress bar
    progress_bar = tqdm(total=len(data_dict), unit="item", desc="Processing items")

    # Define a function to process a single item and update the progress bar
    def process_item(key, value):
        if isinstance(value, list):
            value = value[0]
        resized_base64 = process_images(value)
        progress_bar.update(1)  # Update the progress bar here
        return key, resized_base64

    with ThreadPoolExecutor(max_workers=256) as executor:
        future_to_key = {executor.submit(process_item, key, value): key for key, value in data_dict.items()}

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                resized_data_dict[key] = future.result()
            except Exception as e:
                print(f"Warning: Failed to process key {key}. Error: {e}")
                dropped_keys.append(key)
                progress_bar.update(1)  # Update the progress bar for failed items as well

    # Close the progress bar after all tasks are done
    progress_bar.close()

    ddf = dd.from_pandas(pd.DataFrame.from_dict(resized_data_dict, orient="index", columns=["base64"]), npartitions=nparitions)
    ddf.to_parquet(output_path, engine="pyarrow")

    end_time = time.time()
    print(f"Converting {input_path} to parquet takes {end_time - start_time} seconds.")
    return dropped_keys


def main():
    parser = argparse.ArgumentParser(description="Convert JSON to Parquet")
    parser.add_argument("--input_path", help="Path to the input JSON file")
    parser.add_argument("--output_path", help="Path for the output Parquet file")
    parser.add_argument("--resize_res", type=int, default=-1)
    parser.add_argument("--max_partition_size_gb", type=float, default=1.5, help="Maximum size of each partition in GB")
    args = parser.parse_args()

    # Convert GB to bytes for max_partition_size
    max_partition_size = args.max_partition_size_gb * 1024**3

    dropped_keys = convert_json_to_parquet(args.input_path, args.output_path, max_partition_size)
    print(f"Number of dropped keys: {len(dropped_keys)}")
    print(f"Dropped keys: {dropped_keys}")

if __name__ == "__main__":
    main()
