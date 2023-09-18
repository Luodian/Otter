cd /home/libo0013_e_ntu_edu_sg/projects/Otter-2

export PYTHONPATH=.

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT


H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
export THEID=$THEID
echo $THEID

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_zero3_slurm.yaml \
    pipeline/train/instruction_following.py \
    --pretrained_model_name_or_path=/home/libo0013_e_ntu_edu_sg/projects/Otter-2/checkpoints/OTTER-MPT7B-Instruct0710 \
    --customized_config=/home/libo0013_e_ntu_edu_sg/projects/Otter-2/shared_scripts/Otter_MPT7B_Train_Decoder_4K.json \
    --model_name=otter \
    --inst_format=simple \
    --training_data_yaml=/home/libo0013_e_ntu_edu_sg/projects/shared_scripts/gcp_cluster_recipe.yaml \
    --batch_size=2 \
    --num_epochs=1 \
    --report_to_wandb \
    --wandb_entity=ntu-slab \
    --external_save_dir=/home/libo0013_e_ntu_edu_sg/projects/Otter-2/checkpoints \
    --run_name=Otter_MPT_0916_GCP_SLURM \
    --wandb_project=Otter_Various_Instructions \
    --workers=8 \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --save_hf_model \
    --save_ckpt_each_epoch \
    --max_seq_len=2040