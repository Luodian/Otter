import json
from tqdm import tqdm
import csv

# cur_file_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/complex_reasoning_77k_image2image.json"
cur_file_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/complex_reasoning_77k_text2text.json"
# cur_file_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/conversation_58k/conversation_58k_new_format.json"
# cur_file_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/detail_23k/detail_23k_new_format.json"


with open(cur_file_path) as f:
    cur_file = json.load(f)

import pdb;pdb.set_trace()

if "conversation" in cur_file_path:
    with open("/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/conversation_58k/conversation_58k_new_format.json") as f:
        conversation_instructs = json.load(f)
    conversation_dict = {}
    for i in conversation_instructs:
        _, conversation_id, round_id = i.split('_')
        if conversation_id not in conversation_dict:
            conversation_dict[conversation_id] = 0
        conversation_dict[conversation_id] = max(int(round_id),conversation_dict[conversation_id])        


target_json_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/complex_reasoning_77k_text2text_train.json"
# target_json_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/complex_reasoning_77k_image2image_train.json"
# target_json_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/conversation_58k/conversation_58k_train.json"
# target_json_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/conversation_58k/conversation_58k_new_format_train.json"


target_json = {}
for cur_id in tqdm(cur_file):
    if "conversation" in target_json_path:
        _, conversation_id, round_id = cur_id.split('_')
        if cur_id == f"Conv_{conversation_id}_{conversation_dict[conversation_id]}":
            cur_info = cur_file[cur_id]
            instruction_id = cur_info["instruction_id"]
            in_context_example_ids = cur_info["in_context_example_ids"]
    else:
        cur_info = cur_file[cur_id]
        instruction_id = cur_info["instruction_id"]
        in_context_example_ids = cur_info["in_context_example_ids"]

    target_json[instruction_id] = {
        "instruction_id": instruction_id,
        "in_context_example_ids": in_context_example_ids,
    }


with open(target_json_path,"w") as f:
    json.dump(target_json,f)


