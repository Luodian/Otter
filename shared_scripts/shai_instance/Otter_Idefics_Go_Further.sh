cd /mnt/petrelfs/libo.p/Otter

export PYTHONPATH=.

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_zero3.yaml \
    pipeline/train/instruction_following.py \
    --pretrained_model_name_or_path=/mnt/petrelfs/share_data/zhangyuanhan/otter/otter_idefics9b_instruct_init/ \
    --training_data_yaml=/home/luodian/projects/Otter/shared_scripts/shai_instance/shai_data_recipe.yaml \
    --model_name=idefics \
    --instruction_format=idefics \
    --batch_size=1 \
    --num_epochs=3 \
    --report_to_wandb \
    --wandb_entity=ntu-slab \
    --external_save_dir=/mnt/petrelfs/libo.p/Otter/checkpoints \
    --run_name=Otter_Idefics_0918_train_full_model \
    --wandb_project=Otter_Various_Instructions \
    --workers=16 \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --save_hf_model \
    --save_ckpt_each_epoch