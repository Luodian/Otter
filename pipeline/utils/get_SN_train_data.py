import json
from tqdm import tqdm
import csv

cur_file_path = "/mnt/petrelfs/zhangyuanhan/data/mimicit/SN/SN_instructions.json"


with open(cur_file_path) as f:
    cur_file = json.load(f)

cur_file = cur_file["data"]


conversation_dict = {}
for cur_id in cur_file:
    # import pdb;pdb.set_trace()
    _, _, _, scene_id, _, activity_id, round_id = cur_id.split("_")
    conversation_id = f"{scene_id}_00_{activity_id}_round"
    if conversation_id not in conversation_dict:
        conversation_dict[conversation_id] = 0
    conversation_dict[conversation_id] = max(
        int(round_id[-1]), conversation_dict[conversation_id]
    )

target_json_path = "/mnt/petrelfs/zhangyuanhan/data/mimicit/SN/SN_train.json"


target_json = {}
max_frame = 0
for cur_id in tqdm(cur_file):
    instruction_id = cur_id
    # SN_00_INS_scene0000_00_activity0_round0
    _, _, _, scene_id, _, activity_id, round_id = cur_id.split("_")
    conversation_id = f"{scene_id}_00_{activity_id}_round"

    if cur_id == f"SN_00_INS_{conversation_id}{conversation_dict[conversation_id]}":
        target_json[instruction_id] = cur_file[cur_id]["rel_ins_ids"]

        max_frame = max(len(cur_file[cur_id]["image_ids"]), max_frame)

print(max_frame)
import pdb

pdb.set_trace()
with open(target_json_path, "w") as f:
    json.dump(target_json, f)
