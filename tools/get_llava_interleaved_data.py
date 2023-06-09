import json
from tqdm import tqdm
import csv

# similarity_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/features/clip_vitb16_img2img-top10-similarity.json"
similarity_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/features/all-MiniLM-L6-v1_text2text-top10-similarity.json"

with open(similarity_path) as f:
    cur_similarity = json.load(f)


cur_file = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k.json"
# cur_file = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/conversation_58k.json"

with open(cur_file) as f:
    cur_file = json.load(f)


target_json_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/complex_reasoning_77k_text2text.json"
# target_json_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/complex_reasoning_77k_image2image.json"
# target_json_path = "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/conversation_58k/conversation_58k.json"

target_json = {}
for cur_dict in tqdm(cur_file):
    if "conversation" in target_json_path:
        for cur_round in range(0,len(cur_dict['conversations']),2):
            real_round = int(cur_round / 2)
            instruction_id = f"Conv_{cur_dict['id']}"
            instruction_id = f"{instruction_id}_{real_round}"
            instruction = cur_dict['conversations'][cur_round]["value"].strip().replace("\n","")#.replace("<image>\n","") #cur_meta[uniq_id]["question"]
            answer = cur_dict['conversations'][cur_round+1]["value"].strip().replace("\n","#").replace('\r','#').replace('\t','#') #cur_meta[uniq_id]["answer"]
            image_id = cur_dict["id"]
            in_context_example_ids = [f"Conv_{cur_dict['id']}_{prompt_id}" for prompt_id in range(real_round)] if real_round != 0 else []
            split = "original"#"in_context"
            dataset_name = "llava_conversation_58k"
            type = "llava_conversation_58k"
    else:
        instruction_id = f"CR_T2T_{cur_dict['id']}"
        if instruction_id in target_json:
            continue
        instruction = cur_dict['conversations'][0]["value"].strip().replace("\n","")#("<image>\n","") #cur_meta[uniq_id]["question"]
        answer = cur_dict['conversations'][1]["value"].strip().replace("\n","#").replace('\r','#').replace('\t','#') #cur_meta[uniq_id]["answer"]
        image_id = cur_dict["id"]
        in_context_example_ids = [f"CR_T2T_{prompt_id}" for prompt_id in cur_similarity[image_id]]
        split = "in_context"
        dataset_name = "llava_complext_reasoning_77k_text2text"
        type = "llava_complext_reasoning_77k_text2text"

    target_json[instruction_id] = {
        "instruction_id": instruction_id,
        "instruction": instruction,
        "answer": answer,
        "image_ids": image_id,
        "in_context_example_ids": in_context_example_ids,
        "split": split,
        "dataset_name": dataset_name,
        "type": type
    }


with open(target_json_path,"w") as f:
    json.dump(target_json,f)


