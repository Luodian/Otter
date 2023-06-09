import json
from tqdm import tqdm
import csv

# cur_file_path = "/mnt/petrelfs/zhangyuanhan/data/mimicit/SD/SD_instructions.json"
cur_file_path = "/mnt/petrelfs/zhangyuanhan/data/mimicit/FunQA/FunQA_instructions.json"

with open(cur_file_path) as f:
    cur_file = json.load(f)

cur_file = cur_file["data"]


target_json_path = "/mnt/petrelfs/zhangyuanhan/data/mimicit/FunQA/FunQA_train.json"


target_json = {}
for cur_id in tqdm(cur_file):
    instruction_id = cur_id

    target_json[instruction_id] = cur_file[cur_id]["rel_ins_ids"]
    # target_json[instruction_id] = []


with open(target_json_path, "w") as f:
    json.dump(target_json, f)
