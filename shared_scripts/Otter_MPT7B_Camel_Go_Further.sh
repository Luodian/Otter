cd /mnt/petrelfs/libo.p/Otter

export PYTHONPATH=.

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_zero3.yaml \
    pipeline/train/instruction_following.py \
    --pretrained_model_name_or_path=/mnt/petrelfs/libo.p/Otter/checkpoints/OTTER-MPT7B-Instruct0710 \
    --customized_config=/mnt/petrelfs/libo.p/Otter/shared_scripts/Otter_MPT7B_Train_Decoder_4K.json \
    --model_name=otter \
    --inst_format=simple \
    --training_data_yaml=/mnt/petrelfs/libo.p/Otter/shared_scripts/data_recipe.yaml \
    --batch_size=1 \
    --num_epochs=1 \
    --report_to_wandb \
    --wandb_entity=ntu-slab \
    --external_save_dir=/mnt/petrelfs/libo.p/Otter/checkpoints \
    --run_name=Otter_MPT_0912 \
    --wandb_project=Otter_Various_Instructions \
    --workers=8 \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --save_hf_model \
    --save_ckpt_each_epoch \
    --max_seq_len=2042