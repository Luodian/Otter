
export PYTHONPATH=. 

python -m torch.distributed.run --nproc_per_node=4 --master_port=29501 collie_core/train/instruction_following_origin.py \
--resume_from_checkpoint=/mnt/petrelfs/zhangyuanhan/weights/openflamingo/checkpoint.pt \
--lm_path=luodian/llama-7b-hf \
--tokenizer_path=luodian/llama-7b-hf \
--dataset_resampled \
--multi_instruct_path=/mnt/petrelfs/zhangyuanhan/data/LLaVA-Instruct-150K/conversation_58k/conversation_58k_nocontext.tsv.shuffle.head100000 \
--batch_size=1 \
--num_epochs=3 \
--report_to_wandb \
--wandb_entity=ntu-slab \
--run_name=multi_instruct_conversation-58k-nocontext_flamingo9B_warmup_lr1e-5_unmask-answer-token \
--wandb_project=multi_instruct_conversation-58k-nocontext_flamingo9B \
--workers=0 \
--cross_attn_every_n_layers=4 \
--lr_scheduler=cosine \
--delete_previous_checkpoint \
--learning_rate=1e-5 \