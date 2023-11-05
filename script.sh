accelerate launch \
--config_file=pipeline/accelerate_configs/accelerate_config_zero2.yaml \
--num_processes=8 \
--main_process_port=25000 \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path=adept/fuyu-8b \
--training_data_yaml=shared_scripts/Demo_Data.yaml \
--model_name=fuyu \
--instruction_format=fuyu \
--batch_size=3 \
--num_epochs=6 \
--wandb_entity=lance777 \
--external_save_dir=./checkpoints \
--save_hf_model \
--run_name=Fuyu-Tester-M3IT \
--wandb_project=Fuyu \
--report_to_wandb \
--workers=4 \
--lr_scheduler=linear \
--learning_rate=2e-5 \
--warmup_steps_ratio=0.01 \
--max_seq_len=1024 \
--image_resolution=418,418






srun --nodes=1 --nodelist=queue1-dy-p5-1  --mpi=pmi2 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=80 --pty /bin/bash

sbatch --wrap "sleep 300"  -G 1



TinyLlama 1.1B, TinyLlama codebase:
A100 40G: 24k tokens/s/GPU
H100: 53k tokens/s/GPU

Fuyu 8B, Otter2 codebase：
H100: 3.5 items/s/GPU