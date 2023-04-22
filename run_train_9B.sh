# source ~/.bashrc

PYTHONPATH=. 

accelerate launch --config_file /mnt/lustre/yhzhang/PET-VLM/accelerate_config.yaml --main_process_port 20688  collie_core/train/instruction_following.py \
    --resume_from_checkpoint=/mnt/lustre/share/yhzhang/azure/models/openflamingo/checkpoint.pt \
    --run_name=flamingo9B \
    --lm_path=decapoda-research/llama-7b-hf \
    --tokenizer_path=decapoda-research/llama-7b-hf \
    --dataset_resampled \
    --multi_instruct_path=/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k_detail_23k.tsv \
    --batch_size=8 \
    --num_epochs=5 \
    --report_to_wandb \
    --wandb_project=flamingo9B \
    --wandb_entity=ntu-slab \
    --delete_previous_checkpoint