import argparse
import base64
import json
import os
import tarfile
import uuid
import sys
import braceexpand
import webdataset as wds

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--output_dir", type=str)
arg_parser.add_argument(
    "--image_shards",
    type=str,
    help="Pass in a list of shards in the format path_to_shard/shard_{0..23098}_images_v2.tar",
)
arg_parser.add_argument(
    "--doc_shards",
    type=str,
    help="Pass in a list of shards in the format path_to_shard/docs_shard_{0..23098}_v2.jsonl",
)
args = arg_parser.parse_args()

from tqdm import tqdm


def main(args, start_number=0):
    os.makedirs(args.output_dir, exist_ok=True)

    doc_shards = list(braceexpand.braceexpand(args.doc_shards))
    image_shards = list(braceexpand.braceexpand(args.image_shards))

    assert len(doc_shards) == len(image_shards), "Each doc shard must have a corresponding image shard"
    with wds.ShardWriter(args.output_dir + f"/%09d.tar", maxcount=30000, maxsize=1e10) as sink:
        for idx in tqdm(range(start_number, len(doc_shards)), desc="Converting shards"):
            try:
                image_tar = tarfile.open(image_shards[idx])
            except Exception as e:
                print(e)
                continue

            # Read the JSONL file
            try:
                with open(doc_shards[idx], "r") as json_file:
                    for sample_data in json_file:
                        # get image names from json
                        sample_data = json.loads(sample_data)
                        image_info = sample_data["image_info"]
                        image_names = [image["image_name"] for image in image_info]

                        # Add each image to the tar file
                        for img_idx, image_name in enumerate(image_names):
                            image = image_tar.extractfile(f"{image_tar.getnames()[0]}/{image_name}")

                            # convert to base64
                            image_bytes = image.read()
                            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                            sample_data["image_info"][img_idx]["image_base64"] = image_base64

                        key_str = uuid.uuid4().hex
                        sink.write({"__key__": key_str, "json": sample_data})
            except Exception as e:
                print(e)
                image_tar.close()
                continue

            image_tar.close()


if __name__ == "__main__":
    main(args=args)
