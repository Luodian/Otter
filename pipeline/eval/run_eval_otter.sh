#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_ADDR="localhost"
export MASTER_PORT="29501"
export WORLD_SIZE=4
export RANK=0

cd ~/Otter

realpath .
# pipeline/eval/evaluate.py
python -m pipeline.eval.evaluate \
    --model=otter \
    --model_path="/mnt/petrelfs/share_data/libo/OTTER-MPT7B-Instruct0710" \
    --checkpoint_path="/mnt/petrelfs/share_data/libo/OTTER-MPT7B-Instruct0710/final_weights.pt" \
    --results_file="OTTER_0725_pub_eval_results.json" \
    --precision="bf16" \
    --batch_size=8 \
    --eval_coco \
    --eval_vqav2 \
    --eval_ok_vqa \
    --eval_textvqa \
    --eval_vizwiz \
    --device="cuda" \
    --coco_train_image_dir_path "/mnt/petrelfs/libo.p/data/coco/images/train2014" \
    --coco_val_image_dir_path "/mnt/petrelfs/libo.p/data/coco/images/val2014" \
    --coco_karpathy_json_path "/mnt/petrelfs/libo.p/data/coco/dataset_coco.json" \
    --coco_annotations_json_path "/mnt/petrelfs/libo.p/data/coco/coco2014_annotations/annotations/captions_val2014.json" \
    --vqav2_train_image_dir_path "/mnt/petrelfs/libo.p/data/coco/images/train2014" \
    --vqav2_train_annotations_json_path "/mnt/petrelfs/libo.p/data/vqav2/annotations/v2_mscoco_train2014_annotations.json" \
    --vqav2_train_questions_json_path "/mnt/petrelfs/libo.p/data/vqav2/annotations/v2_OpenEnded_mscoco_train2014_questions.json" \
    --vqav2_test_image_dir_path "/mnt/petrelfs/libo.p/data/coco/images/val2014" \
    --vqav2_test_annotations_json_path "/mnt/petrelfs/libo.p/data/vqav2/annotations/v2_mscoco_val2014_annotations.json" \
    --vqav2_test_questions_json_path "/mnt/petrelfs/libo.p/data/vqav2/annotations/v2_OpenEnded_mscoco_val2014_questions.json" \
    --ok_vqa_train_image_dir_path "/mnt/petrelfs/libo.p/data/okvqa/images/train2014" \
    --ok_vqa_train_annotations_json_path "/mnt/petrelfs/libo.p/data/okvqa/annotations/mscoco_train2014_annotations.json" \
    --ok_vqa_train_questions_json_path "/mnt/petrelfs/libo.p/data/okvqa/annotations/OpenEnded_mscoco_train2014_questions.json" \
    --ok_vqa_test_image_dir_path "/mnt/petrelfs/libo.p/data/okvqa/images/val2014" \
    --ok_vqa_test_annotations_json_path "/mnt/petrelfs/libo.p/data/okvqa/annotations/mscoco_val2014_annotations.json" \
    --ok_vqa_test_questions_json_path "/mnt/petrelfs/libo.p/data/okvqa/annotations/OpenEnded_mscoco_val2014_questions.json" \
    --textvqa_image_dir_path "/mnt/petrelfs/libo.p/data/textvqa/images/train_images/" \
    --textvqa_train_questions_json_path "/mnt/petrelfs/libo.p/data/textvqa/annotations/train_questions_vqa_format.json" \
    --textvqa_train_annotations_json_path "/mnt/petrelfs/libo.p/data/textvqa/annotations/train_annotations_vqa_format.json" \
    --textvqa_test_questions_json_path "/mnt/petrelfs/libo.p/data/textvqa/annotations/val_questions_vqa_format.json" \
    --textvqa_test_annotations_json_path "/mnt/petrelfs/libo.p/data/textvqa/annotations/val_annotations_vqa_format.json" \
    --vizwiz_train_image_dir_path "/mnt/petrelfs/libo.p/data/vizwiz/train" \
    --vizwiz_test_image_dir_path "/mnt/petrelfs/libo.p/data/vizwiz/val" \
    --vizwiz_train_questions_json_path "/mnt/petrelfs/libo.p/data/vizwiz/annotations/train_questions_vqa_format.json" \
    --vizwiz_train_annotations_json_path "/mnt/petrelfs/libo.p/data/vizwiz/annotations/train_annotations_vqa_format.json" \
    --vizwiz_test_questions_json_path "/mnt/petrelfs/libo.p/data/vizwiz/annotations/val_questions_vqa_format.json" \
    --vizwiz_test_annotations_json_path "/mnt/petrelfs/libo.p/data/vizwiz/annotations/val_annotations_vqa_format.json"
# --hateful_memes_image_dir_path "/mnt/petrelfs/libo.p/data/hateful_memes/img" \
# --hateful_memes_train_annotations_json_path "/mnt/petrelfs/libo.p/data/hateful_memes/train.json" \
# --hateful_memes_test_annotations_json_path "/mnt/petrelfs/libo.p/data/hateful_memes/dev.json" \
# --flickr_image_dir_path "/path/to/flickr30k/flickr30k-images" \
# --flickr_karpathy_json_path "/mnt/petrelfs/libo.p/data/flickr30k/dataset_flickr30k.json" \
# --flickr_annotations_json_path "/path/to/flickr30k/dataset_flickr30k_coco_style.json" \