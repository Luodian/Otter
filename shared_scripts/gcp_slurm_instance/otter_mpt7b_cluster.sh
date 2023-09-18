cd /home/libo0013_e_ntu_edu_sg/projects/Otter-2

export PYTHONPATH=.

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export NCCL_NET=IB

echo go $COUNT_NODE
echo $HOSTNAMES

echo COUNT_NODE=$COUNT_NODE
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT


H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
export THEID=$THEID

accelerate launch --machine_rank $THEID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --num_processes=32 --num_machines=4 \
    --config_file=./pipeline/accelerate_configs/accelerate_config_zero3_slurm.yaml \
    pipeline/train/instruction_following.py \
    --pretrained_model_name_or_path=/home/libo0013_e_ntu_edu_sg/projects/Otter-2/checkpoints/OTTER-MPT7B-Instruct0710 \
    --customized_config=/home/luodian/projects/Otter/shared_scripts/Otter_MPT7B_Train_Decoder_Triton.json \
    --training_data_yaml=/home/luodian/projects/Otter/shared_scripts/gcp_slurm_instance/gcp_cluster_recipe.yaml \
    --model_name=otter \
    --inst_format=simple \
    --batch_size=1 \
    --num_epochs=6 \
    --report_to_wandb \
    --wandb_entity=ntu-slab \
    --external_save_dir=/home/libo0013_e_ntu_edu_sg/projects/Otter-2/checkpoints \
    --run_name=Otter_MPT_0918_GCP_SLURM \
    --wandb_project=Otter_Various_Instructions \
    --workers=32 \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --save_hf_model \
    --save_ckpt_each_epoch \
    --max_seq_len=2048