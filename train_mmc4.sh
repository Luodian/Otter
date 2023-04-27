

export PYTHONPATH=. 

python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 open_flamingo/train/train.py \
--lm_path facebook/opt-1.3b \
--tokenizer_path facebook/opt-1.3b \
--dataset_resampled \
--mmc4_shards=/mnt/petrelfs/zhangyuanhan/data/mmc4/000000000.tar \
--batch_size_mmc4 1 \
--train_num_samples_mmc4 8 \
--train_num_samples_laion 8 \
--loss_multiplier_laion 0.2 \
--workers=1 \
--num_epochs 250 \
--lr_scheduler constant \
--warmup_steps 5000 \
--mmc4_textsim_threshold 0.32