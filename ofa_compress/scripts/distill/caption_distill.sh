
# clusters
worker_cnt=1
gpus_per_node=1
memory=400000
cpu=4000

# data
batch_size=8
selected_cols=0,4,2
max_src_length=80
max_tgt_length=20
patch_image_size=480

# optimization
lr=2e-05
clip_grad=1.0
schedule='polynomial_decay'
label_smoothing=0.1
weight_decay=0.01

# generation
beam=5
max_len_a=0
max_len_b=16
no_repeat_ngram_size=3

# distill config
kd_loss_weight=10000
kd_loss_type=ce_with_mask
intermediate_matches="first:attention_mse_sum:encoder,first:attention_mse_sum:decoder"

# save
student_model_config=ofa-tiny
load_student_model="ofa-tiny"
teacher_model_path="/home/xxx/xxx/ofa/teacher_models/ofa-large"
task=caption_stage1
eval_cider_cached=/home/xxx/xxx/ofa/cider_cached_tokens/coco-valid-words.p

save=/home/xxx/xxx/ofa/distill/

data_dir=/home/xxx/xxx/ofa/dataset/caption_data
DATA=${data_dir}/caption_stage1_train.tsv,${data_dir}/caption_val.tsv


ckpt_frequency=10
init_method="load_pretrain"
student_model_config="ofa-tiny"

export CUDA_VISIBLE_DEVICES=1

python -m torch.distributed.launch --nproc_per_node=${worker_cnt} --master_port=8488 main_distill.py \
       --generator-version=fairseq \
       --tables=${DATA} \
       --selected-cols=${selected_cols}\
       --task=${task} \
       --schedule=${schedule} \
       --label-smoothing=${label_smoothing} \
       --kd-loss-weight=${kd_loss_weight} \
       --kd-loss-type=${kd_loss_type} \
       --intermediate-matches=${intermediate_matches} \
       --max-src-length=${max_src_length} \
       --max-tgt-length=${max_tgt_length} \
       --patch-image-size=${patch_image_size} \
       --eval-cider-cached-tokens=${eval_cider_cached} \
       --beam=${beam} \
       --max-len-a=${max_len_a} \
       --max-len-b=${max_len_b} \
       --no-repeat-ngram-size=${no_repeat_ngram_size} \
       --weight-decay=${weight_decay} \
       --clip-grad=${clip_grad} \
       --lr=${lr} \
       --batch-size=${batch_size} \
       --init-method=${init_method}  \
       --student-model-config=${student_model_config} \
       --micro-batch-size=${batch_size} \
       --num-epochs=5 \
       --best-score=10e10 \
       --metric=cider \
       --do-train\
       --do-predict\
       --ckpt-frequency=${ckpt_frequency} \
       --output-dir=${save} \
       --load-teacher-model=${teacher_model_path} \
       --load-student-model=${load_student_model} \
       --worker-cnt=${worker_cnt} \
       --gpus-per-node=${gpus_per_node}