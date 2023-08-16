#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1

<<com
Example Slurm evaluation script. 
Notes:
- VQAv2 test-dev and test-std annotations are not publicly available. 
  To evaluate on these splits, please follow the VQAv2 instructions and submit to EvalAI.
  This script will evaluate on the val split.
com

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

echo go $COUNT_NODE
echo $HOSTNAMES
export PYTHONPATH="$PYTHONPATH:Otter"
srun --cpu_bind=v --accel-bind=gn
python -m pipeline.eval.evaluate \
    --model=otter \
    --model_path=/data/bli/checkpoints/OTTER-Image-MPT7B \
    --checkpoint_path="/data/bli/checkpoints/OTTER-Image-MPT7B/final_weights.pt" \
    --results_file="OTTER_0725_pub_eval_results.json" \
    --precision="bf16" \
    --batch_size=8 \
    --eval_coco \
    --eval_vqav2 \
    --eval_ok_vqa \
    --eval_textvqa \
    --eval_vizwiz \
    --device="cuda" \
    --coco_train_image_dir_path "/data/pufanyi/download/lavis/coco/images/train2014" \
    --coco_val_image_dir_path "/data/pufanyi/download/lavis/coco/images/val2014" \
    --coco_karpathy_json_path "/data/pufanyi/download/lavis/coco/dataset_coco.json" \
    --coco_annotations_json_path "/data/pufanyi/download/lavis/coco/coco2014_annotations/annotations/captions_val2014.json" \
    --vqav2_train_image_dir_path "/data/pufanyi/download/lavis/coco/images/train2014" \
    --vqav2_train_annotations_json_path "/data/pufanyi/download/lavis/vqav2/annotations/v2_mscoco_train2014_annotations.json" \
    --vqav2_train_questions_json_path "/data/pufanyi/download/lavis/vqav2/annotations/v2_OpenEnded_mscoco_train2014_questions.json" \
    --vqav2_test_image_dir_path "/data/pufanyi/download/lavis/coco/images/val2014" \
    --vqav2_test_annotations_json_path "/data/pufanyi/download/lavis/vqav2/annotations/v2_mscoco_val2014_annotations.json" \
    --vqav2_test_questions_json_path "/data/pufanyi/download/lavis/vqav2/annotations/v2_OpenEnded_mscoco_val2014_questions.json" \
    --ok_vqa_train_image_dir_path "/data/pufanyi/download/lavis/okvqa/images/train2014" \
    --ok_vqa_train_annotations_json_path "/data/pufanyi/download/lavis/okvqa/annotations/mscoco_train2014_annotations.json" \
    --ok_vqa_train_questions_json_path "/data/pufanyi/download/lavis/okvqa/annotations/OpenEnded_mscoco_train2014_questions.json" \
    --ok_vqa_test_image_dir_path "/data/pufanyi/download/lavis/okvqa/images/val2014" \
    --ok_vqa_test_annotations_json_path "/data/pufanyi/download/lavis/okvqa/annotations/mscoco_val2014_annotations.json" \
    --ok_vqa_test_questions_json_path "/data/pufanyi/download/lavis/okvqa/annotations/OpenEnded_mscoco_val2014_questions.json" \
    --textvqa_image_dir_path "/data/pufanyi/download/lavis/textvqa/images/train_images/" \
    --textvqa_train_questions_json_path "/data/pufanyi/download/lavis/textvqa/annotations/train_questions_vqa_format.json" \
    --textvqa_train_annotations_json_path "/data/pufanyi/download/lavis/textvqa/annotations/train_annotations_vqa_format.json" \
    --textvqa_test_questions_json_path "/data/pufanyi/download/lavis/textvqa/annotations/val_questions_vqa_format.json" \
    --textvqa_test_annotations_json_path "/data/pufanyi/download/lavis/textvqa/annotations/val_annotations_vqa_format.json" \
    --vizwiz_train_image_dir_path "/data/pufanyi/download/lavis/vizwiz/train" \
    --vizwiz_test_image_dir_path "/data/pufanyi/download/lavis/vizwiz/val" \
    --vizwiz_train_questions_json_path "/data/pufanyi/download/lavis/vizwiz/annotations/train_questions_vqa_format.json" \
    --vizwiz_train_annotations_json_path "/data/pufanyi/download/lavis/vizwiz/annotations/train_annotations_vqa_format.json" \
    --vizwiz_test_questions_json_path "/data/pufanyi/download/lavis/vizwiz/annotations/val_questions_vqa_format.json" \
    --vizwiz_test_annotations_json_path "/data/pufanyi/download/lavis/vizwiz/annotations/val_annotations_vqa_format.json"
# --hateful_memes_image_dir_path "/data/pufanyi/download/lavis/hateful_memes/img" \
# --hateful_memes_train_annotations_json_path "/data/pufanyi/download/lavis/hateful_memes/train.json" \
# --hateful_memes_test_annotations_json_path "/data/pufanyi/download/lavis/hateful_memes/dev.json" \
# --flickr_image_dir_path "/path/to/flickr30k/flickr30k-images" \
# --flickr_karpathy_json_path "/data/pufanyi/download/lavis/flickr30k/dataset_flickr30k.json" \
# --flickr_annotations_json_path "/path/to/flickr30k/dataset_flickr30k_coco_style.json" \
