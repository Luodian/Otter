#!/bin/bash
cd /root/of/Otter

export PYTHONPATH=.

# sent to sub script
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12955
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export NCCL_NET=IB

echo HOSTNAMES = $HOSTNAMES
echo hostname = $(hostname)
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

GPU=$((${COUNT_NODE} * 8))
WORKERS=$((${COUNT_NODE} * 8))

if [ $WORKERS -gt 112 ]; then
    WORKERS=112
fi

RUN_NAME="RunNamePlaceHolder"

echo GPU=${GPU}
echo COUNT_NODE=$COUNT_NODE
echo WORKERS=8
echo "Running ${RUN_NAME}"

H=$(hostname)
THEID=$(echo -e $HOSTNAMES | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]")
export THEID=$THEID
echo $THEID

pkill python


accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_zero2.yaml \
    --machine_rank $THEID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --num_machines=${COUNT_NODE} --num_processes=${GPU} \
    pipeline/train/instruction_following.py \
    --pretrained_model_name_or_path=adept/fuyu-8b \
    --training_data_yaml=./Demo_Data.yaml \
    --model_name=fuyu \
    --instruction_format=fuyu \
    --batch_size=8 \
    --gradient_accumulation_steps=2 \
    --num_epochs=3 \
    --report_to_wandb \
    --wandb_entity=libo0013 \
    --external_save_dir=./checkpoints \
    --run_name=${RUN_NAME} \
    --wandb_project=Fuyu \
    --workers=${WORKERS} \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.03 \
    --save_hf_model \
    --max_seq_len=1024 \
    --logging_steps=1000 \
    --keep_symbols \
    --save_ckpt_each_epoch \
    --dynamic_resolution \
    --with_task_description
