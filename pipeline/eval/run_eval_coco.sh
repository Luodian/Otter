#!/bin/bash

# Set environment variables
export PYTHONPATH=../..:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

# Run the Python script with the specified arguments
python -m pipeline.eval.evaluate \
    --vqav2_train_image_dir_path=/data/pufanyi/download/lavis/coco/images/train2014 \
    --vqav2_train_annotations_json_path=/data/pufanyi/download/lavis/vqav2/annotations/v2_mscoco_train2014_annotations.json \
    --vqav2_train_questions_json_path=/data/pufanyi/download/lavis/vqav2/annotations/v2_OpenEnded_mscoco_train2014_questions.json \
    --vqav2_test_image_dir_path=/data/pufanyi/download/lavis/coco/images/val2014 \
    --vqav2_test_annotations_json_path=/data/pufanyi/download/lavis/vqav2/annotations/v2_mscoco_val2014_annotations.json \
    --vqav2_test_questions_json_path=/data/pufanyi/download/lavis/vqav2/annotations/v2_OpenEnded_mscoco_val2014_questions.json \
    --model=otter \
    --model_path=/data/bli/checkpoints/OTTER-Image-MPT7B \
    --checkpoint_path=/data/bli/checkpoints/OTTER-Image-MPT7B/final_weights.pt \
    --device_map=auto \
    --precision=fp32 \
    --batch_size=8 \
    --eval_vqav2



#!/bin/bash

# export PYTHONPATH=../..:$PYTHONPATH
# python evaluate.py \
#     --model_path "/data/bli/checkpoints/OTTER-Image-MPT7B" \
#     --checkpoint_path "/data/bli/checkpoints/OTTER-Image-MPT7B/final_weights.pt" \
#     --device_map "auto" \
#     --precision fp32 \
#     --batch_size 8 \
#     --eval_vqav2 \
#     --vqav2_train_image_dir_path "/data/666/download/lavis/coco/images/train2014" \
#     --vqav2_train_annotations_json_path "/data/666/download/lavis/vqav2/annotations/v2_mscoco_train2014_annotations.json" \
#     --vqav2_train_questions_json_path "/data/666/download/lavis/vqav2/annotations/v2_OpenEnded_mscoco_train2014_questions.json" \
#     --vqav2_test_image_dir_path "/data/666/download/lavis/coco/images/val2014" \
#     --vqav2_test_annotations_json_path "/data/666/download/lavis/vqav2/annotations/v2_mscoco_val2014_annotations.json" \
#     --vqav2_test_questions_json_path "/data/666/download/lavis/vqav2/annotations/v2_OpenEnded_mscoco_val2014_questions.json" \
    # --flickr_image_dir_path "/path/to/flickr30k/flickr30k-images" \
    # --flickr_karpathy_json_path "/path/to/flickr30k/dataset_flickr30k.json" \
    # --flickr_annotations_json_path "/path/to/flickr30k/dataset_flickr30k_coco_style.json" \
    # --ok_vqa_train_image_dir_path "/path/to/okvqa/train2014" \
    # --ok_vqa_train_annotations_json_path "/path/to/okvqa/mscoco_train2014_annotations.json" \
    # --ok_vqa_train_questions_json_path "/path/to/okvqa/OpenEnded_mscoco_train2014_questions.json" \
    # --ok_vqa_test_image_dir_path "/path/to/okvqa/val2014" \
    # --ok_vqa_test_annotations_json_path "/path/to/okvqa/mscoco_val2014_annotations.json" \
    # --ok_vqa_test_questions_json_path "/path/to/okvqa/OpenEnded_mscoco_val2014_questions.json" \
    # --textvqa_image_dir_path "/path/to/textvqa/train_images/" \
    # --textvqa_train_questions_json_path "/path/to/textvqa/train_questions_vqa_format.json" \
    # --textvqa_train_annotations_json_path "/path/to/textvqa/train_annotations_vqa_format.json" \
    # --textvqa_test_questions_json_path "/path/to/textvqa/val_questions_vqa_format.json" \
    # --textvqa_test_annotations_json_path "/path/to/textvqa/val_annotations_vqa_format.json" \
    # --vizwiz_train_image_dir_path "/path/to/v7w/train" \
    # --vizwiz_test_image_dir_path "/path/to/v7w/val" \
    # --vizwiz_train_questions_json_path "/path/to/v7w/train_questions_vqa_format.json" \
    # --vizwiz_train_annotations_json_path "/path/to/v7w/train_annotations_vqa_format.json" \
    # --vizwiz_test_questions_json_path "/path/to/v7w/val_questions_vqa_format.json" \
    # --vizwiz_test_annotations_json_path "/path/to/v7w/val_annotations_vqa_format.json" \
    # --hateful_memes_image_dir_path "/path/to/hateful_memes/img" \
    # --hateful_memes_train_annotations_json_path "/path/to/hateful_memes/train.json" \
    # --hateful_memes_test_annotations_json_path "/path/to/hateful_memes/dev.json" \





    # --vision_encoder_path ViT-L-14 \
    # --vision_encoder_pretrained openai\
    # --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    # --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    # --cross_attn_every_n_layers 1 \
    # --results_file "results.json" \
    # --precision amp_bf16 \