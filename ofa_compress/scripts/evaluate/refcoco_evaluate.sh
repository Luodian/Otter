
# clusters
worker_cnt=1
gpus_per_node=1
memory=400000
cpu=3000

# data
batch_size=8
selected_cols=0,4,2,3
max_src_length=80
max_tgt_length=30
patch_image_size=512
constraint_range=58457,59457

# generation
beam=5
max_len_a=0
max_len_b=4
min_len=4

# save

task=refcoco

save=/home/xxx/xxx/ofa/evaluate/

data_dir=/home/xxx/xxx/ofa/dataset/refcoco_data
DATA=${data_dir}/refcoco_train.tsv,${data_dir}/refcoco_val.tsv
load=/home/xxx/xxx/ofa/results_distilled/refcoco


ckpt_frequency=10

export CUDA_VISIBLE_DEVICES=1

python -m torch.distributed.launch --nproc_per_node=${worker_cnt} --master_port=9999 main_evaluate.py \
       --load=${load} \
       --tables=${DATA} \
       --selected-cols=${selected_cols}\
       --task=${task} \
       --max-src-length=${max_src_length} \
       --max-tgt-length=${max_tgt_length} \
       --patch-image-size=${patch_image_size} \
       --constraint-range=${constraint_range} \
       --beam=${beam} \
       --max-len-a=${max_len_a} \
       --max-len-b=${max_len_b} \
       --min-len=${min_len} \
       --max-src-length=${max_src_length} \
       --max-tgt-length=${max_tgt_length} \
       --batch-size=${batch_size} \
       --micro-batch-size=${batch_size} \
       --num-epochs=5 \
       --best-score=10e10 \
       --metric=ap \
       --do-predict\
       --ckpt-frequency=${ckpt_frequency} \
       --output-dir=${save} \
       --worker-cnt=${worker_cnt} \
       --gpus-per-node=${gpus_per_node}
