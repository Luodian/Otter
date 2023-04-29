
export PYTHONPATH=. 

python -m torch.distributed.run --nproc_per_node=4 pipeline/train/instruction_following.py \
--pretrained_model_name_or_path=/mnt/lustre/yhzhang/weights/openflamingo_9b_hf \
--lm_path=/mnt/lustre/yhzhang/weights/llama-7b-hf \
--tokenizer_path=/mnt/lustre/yhzhang/weights/llama-7b-hf \
--dataset_resampled \
--multi_instruct_path=/mnt/lustre/yhzhang/data/LLaVA-Instruct-150K/complex_reasoning_77k_detail_23k.tsv.head8 \
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
--learning_rate=1e-5 \