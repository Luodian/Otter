

PYTHONPATH=. 
# LD_LIBRARY_PATH=/mnt/lustre/share/cuda-11.7/lib64

accelerate launch --config_file /mnt/lustre/yhzhang/PET-VLM/accelerate_config_zero3.yaml collie_core/train/instruction_following.py \
    --run_name=flamingo3B \
    --lm_path=facebook/opt-1.3b \
    --tokenizer_path=facebook/opt-1.3b \
    --dataset_resampled \
    --multi_instruct_path=/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k_detail_23k.tsv.head8 \
    --batch_size=8 \
    --num_epochs=1 \
    --report_to_wandb \
    --wandb_project=flamingo3B \
    --wandb_entity=ntu-slab \
    --delete_previous_checkpoint \
