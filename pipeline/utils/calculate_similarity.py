import numpy as np
import scipy.spatial.distance as distance
import os
from tqdm import tqdm
import sys
from scipy import linalg, mat, dot
import json
import torch

model_name = "clip_vitb16"  # sys.argv[1]
# source_split = sys.argv[1]
# target_split = sys.argv[2]
source_modal = sys.argv[1]
target_modal = sys.argv[2]

# print(f"Processing {features_name} ...")
# sys.stdout.flush()

# source_dir = "/mnt/lustre/yhzhang/data/ofa/metaicl_data/features"
# source_features_name = f"{source_dir}/coco_{model_name}_caption_{source_split}_features.npz"
# target_features_name = f"{source_dir}/coco_{model_name}_caption_{target_split}_features.npz"

source_dir = (
    "/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/features"
)
source_features_name = f"{source_dir}/{model_name}_features.npz"
target_features_name = f"{source_dir}/{model_name}_features.npz"

source_path = source_features_name
target_path = target_features_name

try:
    source_file_npz = np.load(source_path)
    target_file_npz = np.load(target_path)
except:
    import pdb

    pdb.set_trace()

source_examples = source_file_npz["uniqids"].tolist()
# import pdb;pdb.set_trace()
# import pdb;pdb.set_trace()
target_examples = target_file_npz["uniqids"].tolist()
# source_features = source_file_npz["features"].astype(np.float32)
# target_features = target_file_npz["features"].astype(np.float32)
if source_modal == "image":
    source_features = source_file_npz["image_features"].astype(np.float32)
elif source_modal == "text":
    source_features = source_file_npz["text_features"].astype(np.float32)
else:
    import pdb

    pdb.set_trace()

# import pdb;pdb.set_trace()
if target_modal == "image":
    target_features = target_file_npz["image_features"].astype(np.float32)
else:
    target_features = target_file_npz["text_features"].astype(np.float32)

target_sample_idx = np.random.choice(
    target_features.shape[0], size=int(target_features.shape[0]), replace=False
)
target_sample_feature = target_features[target_sample_idx, :]

source_features = torch.from_numpy(source_features).cuda()

target_sample_feature = torch.from_numpy(target_sample_feature).cuda()

similarity_idx_dict = {}

similarity_idx = torch.tensor([]).cuda()

for start_idx in tqdm(range(0, len(source_features), 1000)):
    cur_source_features = source_features[start_idx : start_idx + 1000]

    cur_similarity = torch.mm(cur_source_features, target_sample_feature.T) / (
        torch.norm(cur_source_features, dim=1, keepdim=True)
        * torch.norm(target_sample_feature, dim=1, keepdim=True).T
    )

    # import pdb;pdb.set_trace()
    cur_similarity_idx = torch.argsort(cur_similarity, dim=1)[:, -50:]

    similarity_idx = torch.cat((similarity_idx, cur_similarity_idx))

for _, (cur_example, cur_similarity) in enumerate(zip(source_examples, similarity_idx)):
    img_name = cur_example
    cur_similarity = cur_similarity.cpu().numpy().astype(np.int32)
    cur_similar_name = list(
        target_examples[target_sample_idx[idx]]
        for idx in cur_similarity[::-1]
        if target_examples[target_sample_idx[idx]] != img_name
    )
    # import pdb;pdb.set_trace()
    cur_similar_name = list(dict.fromkeys(cur_similar_name))
    # import pdb;pdb.set_trace()
    assert (
        len(cur_similar_name) >= 10
    ), "num of cur_similar_name is too small, please enlarge the similarity_idx size"

    # import pdb;pdb.set_trace()
    if img_name not in similarity_idx_dict:
        similarity_idx_dict[img_name] = cur_similar_name[:10]

if source_modal == "image":
    if target_modal == "text":
        with open(
            f"{source_dir}/{model_name}_img2text-top10-similarity.json", "w"
        ) as outfile:
            json.dump(similarity_idx_dict, outfile)
    else:
        with open(
            f"{source_dir}/{model_name}_img2img-top10-similarity.json", "w"
        ) as outfile:
            json.dump(similarity_idx_dict, outfile)
else:
    if target_modal == "text":
        with open(
            f"{source_dir}/{model_name}_text2text-top10-similarity.json", "w"
        ) as outfile:
            json.dump(similarity_idx_dict, outfile)
    else:
        with open(
            f"{source_dir}/{model_name}_text2img-top10-similarity.json", "w"
        ) as outfile:
            json.dump(similarity_idx_dict, outfile)
