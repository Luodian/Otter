import json
from tqdm import tqdm
import csv

# similarity_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/features/clip_vitb16_img2img-top10-similarity.json"
# similarity_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/features/all-MiniLM-L6-v1_text2text-top10-similarity.json"

# with open(similarity_path) as f:
#     cur_similarity = json.load(f)


# cur_file = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k.json"
cur_file = "/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/conversation_58k.json"
# cur_file = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/detail_23k.json"

with open(cur_file) as f:
    cur_file = json.load(f)


target_json_path = (
    "/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LACONV_instructions.json"
)
# target_json_path = "/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LACR_I2I_instructions.json"
# target_json_path = "/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LACR_T2T_instructions.json"
# target_json_path = "/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/LA/LADD_instructions.json"


target_json = {}
target_json["meta"] = {"version": "0.0.1", "time": "2023-06", "author": "ntu"}

target_json["data"] = {}

for cur_dict in tqdm(cur_file):
    if "CONV" in target_json_path:
        for cur_round in range(0, len(cur_dict["conversations"]), 2):
            real_round = int(cur_round / 2)
            instruction_id = (
                f"LACONV_00_INS_{cur_dict['id']}"  # LACONV_00_INS_000000033471_2
            )
            instruction_id = f"{instruction_id}_{real_round}"
            instruction = (
                cur_dict["conversations"][cur_round]["value"]
                .strip()
                .replace("<image>", "")
            )  # .replace("\n","")#.replace("<image>\n","") #cur_meta[uniq_id]["question"]
            answer = (
                cur_dict["conversations"][cur_round + 1]["value"]
                .strip()
                .replace("<image>", "")
            )  # .replace("\n","#").replace('\r','#').replace('\t','#') #cur_meta[uniq_id]["answer"]
            image_id = f"LA_00_IMG_{cur_dict['id']}"
            in_context_example_ids = (
                [
                    f"LACONV_00_INS_{cur_dict['id']}_{prompt_id}"
                    for prompt_id in range(real_round)
                ]
                if real_round != 0
                else []
            )

            target_json["data"][instruction_id] = {
                "instruction": instruction,
                "answer": answer,
                "image_ids": [image_id],
                "rel_ins_ids": in_context_example_ids,
            }
    else:
        # instruction_id = f"DD_{cur_dict['id']}"
        # instruction_id = f"CR_T2T_{cur_dict['id']}"
        instruction_id = f"LACR_I2I_00_INS_{cur_dict['id']}"
        if instruction_id in target_json:
            continue
        instruction = (
            cur_dict["conversations"][0]["value"].strip().replace("<image>", "")
        )  # .replace("\n","")#("<image>\n","") #cur_meta[uniq_id]["question"]
        answer = (
            cur_dict["conversations"][1]["value"].strip().replace("<image>", "")
        )  # .replace("\n","#").replace('\r','#').replace('\t','#') #cur_meta[uniq_id]["answer"]
        image_id = cur_dict["id"]
        in_context_example_ids = []
        in_context_example_ids = [
            f"LACR_I2I_00_INS_{prompt_id}" for prompt_id in cur_similarity[image_id]
        ]
        # in_context_example_ids = [f"LACR_T2T_00_INS_{prompt_id}" for prompt_id in cur_similarity[image_id]]
        # dataset_name = "LlavaDetailDescription"
        # dataset_name = "LlavaComplexReasoningT2T"
        # dataset_name = "LlavaComplexReasoningI2I"

        target_json["data"][instruction_id] = {
            "instruction": instruction,
            "answer": answer,
            "image_ids": [image_id],
            "rel_ins_ids": in_context_example_ids,
        }


with open(target_json_path, "w") as f:
    json.dump(target_json, f)
