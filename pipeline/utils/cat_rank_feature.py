import numpy as np

# save_name = f"/mnt/lustre/yhzhang/data/ofa/metaicl_data/features/coco_clip_vitb16_caption_train_features"
save_name = f"/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/features/all-MiniLM-L6-v1_features"


for idx in range(3):
    source_path = f"all-MiniLM-L6-v1_features.rank_{idx}.npz"
    source_file_npz = np.load(
        f"/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k/features/{source_path}"
    )
    # import pdb;pdb.set_trace()
    cur_uniq_id = source_file_npz["uniqids"]
    # cur_img_feature = source_file_npz["image_features"]
    cur_text_feature = source_file_npz["text_features"]
    # cur_text_with_answer_features = source_file_npz["text_with_answer_features"]
    # import pdb;pdb.set_trace()
    if idx == 0:
        uniq_ids = cur_uniq_id
        # imgs_features = cur_img_feature
        text_features = cur_text_feature
        # text_with_answer_features = cur_text_with_answer_features
    uniq_ids = np.concatenate((uniq_ids, cur_uniq_id))
    # imgs_features = np.concatenate((imgs_features, cur_img_feature))
    text_features = np.concatenate((text_features, cur_text_feature))
    # text_with_answer_features = np.concatenate((text_with_answer_features, cur_text_with_answer_features))
# import pdb;pdb.set_trace()

# np.savez(f"{save_name}", uniqids=uniq_ids, features=features)
# np.savez(f"{save_name}", uniqids=uniq_ids, image_features=imgs_features,text_features=text_features)
# np.savez(f"{save_name}", uniqids=uniq_ids, image_features=imgs_features,text_features=text_features, text_with_answer_features=text_with_answer_features)
np.savez(f"{save_name}", uniqids=uniq_ids, text_features=text_features)
