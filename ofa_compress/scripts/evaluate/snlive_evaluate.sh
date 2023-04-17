
# clusters
worker_cnt=1
gpus_per_node=1
memory=400000
cpu=3000

# data
batch_size=8
selected_cols=0,2,3,4,6
max_src_length=80
max_tgt_length=20
patch_image_size=480
prompt_type='prev_output'


# model config


# generation


# save
task=snli_ve

save=/home/xxx/xxx/ofa/evaluate/

data_dir=../../dataset/snli_ve_data
DATA=${data_dir}/snli_ve_train.tsv,${data_dir}/snli_ve_dev.tsv
load=/home/xxx/xxx/ofa/results_distilled/snli_ve



ckpt_frequency=10


python -m torch.distributed.launch --nproc_per_node=${worker_cnt} --master_port=9999 main_evaluate.py \
       --load=${load} \
       --tables=${DATA} \
       --selected-cols=${selected_cols}\
       --add-caption=True \
       --task=${task} \
       --max-src-length=${max_src_length} \
       --max-tgt-length=${max_tgt_length} \
       --patch-image-size=${patch_image_size} \
       --prompt-type=${prompt_type} \
       --batch-size=${batch_size} \
       --micro-batch-size=${batch_size} \
       --num-epochs=5 \
       --best-score=10e10 \
       --metric=acc \
       --do-predict\
       --ckpt-frequency=${ckpt_frequency} \
       --output-dir=${save} \
       --worker-cnt=${worker_cnt} \
       --gpus-per-node=${gpus_per_node} \
