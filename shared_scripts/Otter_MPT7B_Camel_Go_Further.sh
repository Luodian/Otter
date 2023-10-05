cd /mnt/petrelfs/libo.p/Otter

export PYTHONPATH=.

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_zero3.yaml \
    pipeline/train/instruction_following.py \
    --pretrained_model_name_or_path=/mnt/petrelfs/share_data/zhangyuanhan/otter/OTTER-MPT7B-Instruct-0725 \
    --customized_config=/mnt/petrelfs/libo.p/Otter/shared_scripts/Otter_MPT7B_Train_Decoder_4K.json \
    --model_name=otter \
    --instruction_format=simple \
    --training_data_yaml=/home/luodian/projects/Otter/shared_scripts/Demo_Data.yaml \
    --batch_size=2 \
    --num_epochs=3 \
    --report_to_wandb \
    --wandb_entity=ntu-slab \
    --external_save_dir=/mnt/petrelfs/libo.p/Otter/checkpoints \
    --run_name=Otter_MPT_0919 \
    --wandb_project=Otter_Various_Instructions \
    --workers=8 \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --save_hf_model \
    --max_seq_len=2048