cd /home/luodian/projects/Otter

export PYTHONPATH=.

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_zero3.yaml \
    --num_processes=2 \
    pipeline/train/instruction_following.py \
    --pretrained_model_name_or_path=/home/luodian/projects/checkpoints/OTTER-MPT7B-Instruct0710 \
    --model_name=otter \
    --instruction_format=simple \
    --training_data_yaml=/home/luodian/projects/Otter/scripts/gcp_local_recipe.yaml \
    --batch_size=2 \
    --num_epochs=3 \
    --report_to_wandb \
    --wandb_entity=ntu-slab \
    --external_save_dir=/home/luodian/projects/checkpoints \
    --run_name=Otter_MPT_0921 \
    --wandb_project=Otter_Various_Instructions \
    --workers=24 \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --save_hf_model \
    --max_seq_len=2048
