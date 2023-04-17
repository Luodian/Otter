
# clusters
worker_cnt=1
gpus_per_node=1
memory=400000
cpu=4000

# data
batch_size=8
selected_cols=0,2,3,4,6
max_src_length=80
max_tgt_length=20
patch_image_size=480
prompt_type='prev_output'


# optimization
lr=6e-05
clip_grad=0.0
schedule='polynomial_decay'
label_smoothing=0.0
weight_decay=0.01

# generation

# distill config
kd_loss_weight=10000.0
kd_loss_type=ce_with_mask
intermediate_matches="first:attention_mse_sum:encoder,first:attention_mse_sum:decoder"

# save
load_student_model="ofa-tiny"
teacher_model_path="/home/xxx/xxx/ofa/teacher_models/ofa-large"
student_model_config=ofa-tiny
task=snli_ve

save=/home/xxx/xxx/ofa/evaluate/

data_dir=../../dataset/snli_ve_data
DATA=${data_dir}/snli_ve_train.tsv,${data_dir}/snli_ve_dev.tsv



ckpt_frequency=5
init_method="load_pretrain"


python -m torch.distributed.launch --nproc_per_node=${worker_cnt} --master_port=9999 main_distill.py \
       --tables=${DATA} \
       --selected-cols=${selected_cols}\
       --add-caption=True \
       --task=${task} \
       --schedule=${schedule} \
       --label-smoothing=${label_smoothing} \
       --intermediate-matches=${intermediate_matches} \
       --kd-loss-weight=${kd_loss_weight} \
       --kd-loss-type=${kd_loss_type} \
       --max-src-length=${max_src_length} \
       --max-tgt-length=${max_tgt_length} \
       --patch-image-size=${patch_image_size} \
       --prompt-type=${prompt_type} \
       --weight-decay=${weight_decay} \
       --clip-grad=${clip_grad} \
       --lr=${lr} \
       --kd-loss-weight=0.0 \
       --batch-size=${batch_size} \
       --init-method=${init_method}  \
       --student-model-config=${student_model_config} \
       --micro-batch-size=${batch_size} \
       --num-epochs=5 \
       --best-score=10e10 \
       --metric=acc \
       --do-train\
       --do-predict\
       --ckpt-frequency=${ckpt_frequency} \
       --output-dir=${save} \
       --load-teacher-model=${teacher_model_path} \
       --load-student-model=${load_student_model} \
       --worker-cnt=${worker_cnt} \
       --gpus-per-node=${gpus_per_node}
