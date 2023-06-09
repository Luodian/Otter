import json
import csv
from tqdm import tqdm

image_dict = {}
target_json_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/llava_images.json"

with open(
    "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/complex_reasoning_77k.tsv"
) as f:
    metas = f.readlines()
    for i in tqdm(metas):
        (
            uniq_id,
            image,
            caption,
            question,
            refs,
            gt_objects,
            dataset_name,
            type,
        ) = i.rstrip("\n").split("\t")
        if uniq_id not in image_dict:
            image_dict[uniq_id] = {"id": uniq_id, "image": image}
            # with open(target_tsv, 'a', newline='') as tsvfile:
            #     writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            #     writer.writerow([uniq_id, image])

with open(
    "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/conversation_58k/conversation_58k.tsv"
) as f:
    metas = f.readlines()
    for i in tqdm(metas):
        (
            uniq_id,
            image,
            caption,
            question,
            refs,
            gt_objects,
            dataset_name,
            type,
        ) = i.rstrip("\n").split("\t")
        uniq_id = uniq_id.split("_")[0]
        if uniq_id not in image_dict:
            image_dict[uniq_id] = {"id": uniq_id, "image": image}
            # image_dict[uniq_id] = 0
            # with open(target_tsv, 'a', newline='') as tsvfile:
            #     writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            #     writer.writerow([uniq_id, image])


with open(
    "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/detail_23k/detail_23k.tsv"
) as f:
    metas = f.readlines()
    for i in tqdm(metas):
        (
            uniq_id,
            image,
            caption,
            question,
            refs,
            gt_objects,
            dataset_name,
            type,
        ) = i.rstrip("\n").split("\t")
        if uniq_id not in image_dict:
            image_dict[uniq_id] = {"id": uniq_id, "image": image}
            # with open(target_tsv, 'a', newline='') as tsvfile:
            #     writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            #     writer.writerow([uniq_id, image])

with open(target_json_path, "w") as f:
    json.dump(image_dict, f)
