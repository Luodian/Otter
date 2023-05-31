import json
from tqdm import tqdm
import csv
import random


rel_ins_ids_num = 2

# cur_file_path = "/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LACR_I2I_instructions.json"
cur_file_path = (
    "/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LACR_T2T_instructions.json"
)


with open(cur_file_path) as f:
    cur_file = json.load(f)

cur_file = cur_file["data"]


if "CONV" in cur_file_path:
    conversation_dict = {}
    for i in cur_file:
        # import pdb;pdb.set_trace()
        _, _, _, conversation_id, round_id = i.split("_")
        if conversation_id not in conversation_dict:
            conversation_dict[conversation_id] = 0
        conversation_dict[conversation_id] = max(
            int(round_id), conversation_dict[conversation_id]
        )


# target_json_path = "/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LACR_I2I_half_train.json"
target_json_path = (
    "/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LACR_T2T_half_train.json"
)


target_json = {}

random.seed(0)

for cur_id in tqdm(cur_file):
    if "CONV" in target_json_path:
        _, _, _, conversation_id, round_id = cur_id.split("_")
        if (
            cur_id
            == f"LACONV_00_INS_{conversation_id}_{conversation_dict[conversation_id]}"
        ):
            instruction_id = cur_id
            if len(cur_file[cur_id]["rel_ins_ids"]) < rel_ins_ids_num:
                if len(cur_file[cur_id]["rel_ins_ids"]) == 0:
                    continue
                rel_ins_ids = cur_file[cur_id]["rel_ins_ids"] * rel_ins_ids_num
                rel_ins_ids = rel_ins_ids[-rel_ins_ids_num:]
            else:
                rel_ins_ids = cur_file[cur_id]["rel_ins_ids"][-rel_ins_ids_num:]

            target_json[instruction_id] = rel_ins_ids
    else:
        if random.randint(0, 1) == 0:
            continue
        instruction_id = cur_id
        if len(cur_file[cur_id]["rel_ins_ids"]) < rel_ins_ids_num:
            if len(cur_file[cur_id]["rel_ins_ids"]) == 0:
                continue
            rel_ins_ids = cur_file[cur_id]["rel_ins_ids"] * rel_ins_ids_num
            rel_ins_ids = rel_ins_ids[-rel_ins_ids_num:]
        else:
            rel_ins_ids = cur_file[cur_id]["rel_ins_ids"][:rel_ins_ids_num]

        target_json[instruction_id] = rel_ins_ids


with open(target_json_path, "w") as f:
    json.dump(target_json, f)
