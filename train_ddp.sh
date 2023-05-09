export PYTHONPATH=.

accelerate launch --config_file=/mnt/petrelfs/zhangyuanhan/Otter/accelerate_configs/accelerate_config_ddp.yaml \
pipeline/train/instruction_following_ddp.py \
--pretrained_model_name_or_path=/mnt/petrelfs/zhangyuanhan/weights/flamingo_9b_hf \
--dataset_resampled \
--multi_instruct_path=/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/complex_reasoning_77k_detail_23k.tsv.shuffle \
--batch_size=1 \
--num_epochs=6 \
--report_to_wandb \
--wandb_entity=ntu-slab \
--run_name=multi_instruct_chunyuan-core_lr1e-5_6epochs \
--wandb_project=multi_instruct_chunyuan-core_otter9B \
--workers=0 \
--cross_attn_every_n_layers=4 \
--lr_scheduler=cosine \
--delete_previous_checkpoint \
--learning_rate=1e-5 