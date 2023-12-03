import pandas as pd
import os
import time
import json
from tqdm import tqdm
import argparse
import orjson

def process_images(base64_str):
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
        img = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
    except Exception as e:
        print(f"Warning: Failed to open image. Error: {e}")
        return None

    if img.mode == "RGBA":
        img = img.convert("RGB")

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    new_base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return new_base64_str


def convert_json_to_parquet(input_path, output_path):
    start_time = time.time()
    with open(input_path, "rb") as f:
        data_dict = orjson.loads(f.read())
    # with open(input_path, "r") as f:
    #     data_dict = json.load(f)

    resized_data_dict = {}
    dropped_keys = []
    for key, value in tqdm(data_dict.items(), desc=f"Processing {input_path}"):
        if isinstance(value, list):
            value = value[0]
        # resized_base64 = process_images(value)
        resized_data_dict[key] = value

    df = pd.DataFrame.from_dict(resized_data_dict, orient="index", columns=["base64"])
    df.to_parquet(output_path, engine="pyarrow")

    end_time = time.time()
    print(f"Converting {input_path} to parquet takes {end_time - start_time} seconds.")
    return dropped_keys


def main():
    parser = argparse.ArgumentParser(description="Convert JSON to Parquet")
    parser.add_argument("--input_path", help="Path to the input JSON file")
    parser.add_argument("--output_path", help="Path for the output Parquet file")
    args = parser.parse_args()

    dropped_keys = convert_json_to_parquet(args.input_path, args.output_path)
    print(dropped_keys)


if __name__ == "__main__":
    main()
