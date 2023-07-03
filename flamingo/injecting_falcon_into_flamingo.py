import os
import torch
from .configuration_flamingo import FlamingoConfig
from .modeling_flamingo import FlamingoForConditionalGeneration

root_dir = os.environ["AZP"]
print(root_dir)


config = FlamingoConfig.from_json_file(".flamingo-falcon-7B.json")
model = FlamingoForConditionalGeneration(config=config)


state_dict_files = [
    f"{root_dir}/otter/checkpoints/falcon-7b/pytorch_model-00001-of-00002.bin",
    f"{root_dir}/otter/checkpoints/falcon-7b/pytorch_model-00002-of-00002.bin",
]

state_dict = {}
for file in state_dict_files:
    state_dict_part = torch.load(file, map_location="cpu")
    state_dict.update(state_dict_part)


state_dict_3 = torch.load("{root_dir}/otter/checkpoints/flamingo_9b_hf/pytorch_model-00004-of-00004.bin", map_location="cpu")
for cur_key in list(state_dict_3.keys()):
    if "vision_encoder" not in cur_key:
        del state_dict_3[cur_key]

_ = model.load_state_dict(
    state_dict_3,
    False,
)
print(_[1])

save_state_dict_1 = {}
for key in state_dict:
    if ".h." in key:
        _, _, layer_num, *remain_names = key.split(".")
        target_key = f"transformer.h.{layer_num}.decoder_layer.{'.'.join(remain_names)}"
    else:
        target_key = key
    save_state_dict_1[f"{target_key}"] = state_dict[key]
_ = model.lang_encoder.load_state_dict(
    save_state_dict_1,
    False,
)
print(_[1])
model.save_pretrained(f"{root_dir}/otter/checkpoints/flamingo-falcon-7b/")
