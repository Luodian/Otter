import json
from PIL import Image
import base64
import os
import pandas as pd
from tqdm import tqdm


def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"Error: {e}")
        return ""


def process_json_conv(input_json_path, output_json_path, output_train_idx_path, output_img_dict_path, split="coco"):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    output_data = {"meta": {"dataset": f"SHAREGPT4V_{split.upper()}", "version": "2023.12.18", "split": "train"}, "data": {}}
    train_idx_data = {}
    image_dict = {}
    counter = 0

    print(f"Processing {split}..")
    pbar = tqdm(total=len(data))
    for entry in data:
        original_id = str(entry["id"]).replace("/", "_")
        conversations = entry["conversations"]

        img_path = entry["image"] if "image" in entry else ""
        identifier = img_path.split("/")[0] if "image" in entry else "text"
        pbar.set_postfix_str(f"Processed {identifier}..")
        pbar.update(1)
        
        if identifier != split:
            continue

        if identifier == "sam":
            img_path = img_path.replace("sam/images", "sam")
        img_path = os.path.join("/raid/bli/data/SHAREGPT4V/ShareGPT4V", img_path)

        identifier = identifier.upper()
        if img_path != "":
            image_base64 = image_to_base64(img_path)
            if image_base64 == "":
                print(f"Error: {img_path} could not be converted to base64")
                continue
            img_key = f"{identifier}_IMG_{original_id}"
            image_dict[img_key] = image_base64

        rel_ins_ids = []
        ins_idx = 0
        last_ins_key = None

        for idx, conversation in enumerate(conversations):
            if conversation["from"] == "human":
                ins_key = f"{identifier}_{original_id}_{counter}_CONV{str(ins_idx).zfill(2)}"

                if ins_key in output_data["data"]:
                    print(f"Error: {ins_key} already exists")
                    continue

                question = conversation["value"].strip().replace("<image>\n", "").replace("<image>", "")
                answer = conversations[idx + 1]["value"] if idx + 1 < len(conversations) else ""

                output_data["data"][ins_key] = {
                    "instruction": question,
                    "answer": answer,
                    "rel_ins_ids": rel_ins_ids.copy(),
                    "image_ids": [img_key] if img_path != "" else [],
                }

                if idx < len(conversations) - 2:  # Excluding the last instruction
                    rel_ins_ids.append(ins_key)
                else:
                    last_ins_key = ins_key

                ins_idx += 1

        if last_ins_key:
            train_idx_data[last_ins_key] = rel_ins_ids.copy()

        counter += 1

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)

    with open(output_train_idx_path, "w") as f:
        json.dump(train_idx_data, f)

    # with open(output_img_dict_path, "w") as f:
    #     json.dump(image_dict, f)
    if len(image_dict) > 0:
        print(f"For {split}, Saving the new JSON and image dictionary to the disk..")
        image_df = pd.DataFrame.from_dict(image_dict, orient="index", columns=["base64"])
        image_df.to_parquet(output_img_dict_path, engine="pyarrow")


if __name__ == "__main__":
    input_json_path = "/raid/bli/data/SHAREGPT4V/ShareGPT4V/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"
    splits_list = ["sam", "wikiart", "share_textvqa", "web-celebrity", "web-landmark"]

    for split in splits_list:
        output_json_path = f"/raid/bli/data/SHAREGPT4V/MIMICIT_Format/SHAREGPT4V_{split.upper()}.json"
        output_train_path = f"/raid/bli/data/SHAREGPT4V/MIMICIT_Format/SHAREGPT4V_{split.upper()}_train.json"
        output_img_dict_path = f"/raid/bli/data/SHAREGPT4V/MIMICIT_Format/SHAREGPT4V_{split.upper()}_Images.parquet"
        process_json_conv(input_json_path, output_json_path, output_train_path, output_img_dict_path, split=split)