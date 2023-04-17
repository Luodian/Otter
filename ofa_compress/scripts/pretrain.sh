
# clusters
worker_cnt=2
gpus_per_node=8
memory=400000
cpu=4000

# data
batch_size=8


selected_cols=0,1,2,3,4,5,6,7
text_selected_cols=0,1
image_selected_cols=0,1,2
detection_selected_cols=0,1,2
max_src_length=80
max_object_length=30
max_tgt_length=30
patch_image_size=256
max_image_size=512
sample_patch_num=-1
eval_cider_cached=/home/xxx/xxx/ofa/cider_cached_tokens/coco-valid-words.p

# optimization
lr=0.0005
lr_end=5e-06
clip_grad=5.0
schedule='polynomial_decay'
label_smoothing=0.0
weight_decay=0.01





# save
student_model_config=ofa-tiny
save=/home/xxx/xxx/ofa/pretrain/${student_model_config}/
ckpt_frequency=35
init_method="random"


data_dir=../../dataset/pretrain_data
neg_sample_dir=${data_dir}/negative_sample
DATA=${data_dir}/vision_language_examples.tsv,${data_dir}/text_examples.tsv,${data_dir}/image_examples.tsv,${data_dir}/detection_examples.tsv


python -m torch.distributed.launch --nproc_per_node=${worker_cnt} --master_port=9999 main_train.py \
       --tables=${DATA} \
       --selected-cols=${selected_cols} \
       --text-selected-cols=${text_selected_cols} \
       --image-selected-cols=${image_selected_cols} \
       --detection-selected-cols=${detection_selected_cols} \
       --neg-sample-dir=${neg_sample_dir} \
       --label-smoothing=${label_smoothing} \
       --batch-size=${batch_size} \
       --max-src-length=${max_src_length} \
       --max-tgt-length=${max_tgt_length} \
       --max-object-length=${max_object_length} \
       --patch-image-size=${patch_image_size} \
       --sample-patch-num=${sample_patch_num} \
       --max-image-size=${max_image_size} \
       --task=pretrain \
       --schedule=${schedule} \
       --eval-cider-cached-tokens=${eval_cider_cached} \
       --init-method=${init_method} \
       --student-model-config=${student_model_config} \
       --micro-batch-size=${batch_size} \
       --num-epochs=50 \
       --best-score=10e10 \
       --metric=loss \
       --lr=${lr} \
       --lr-end=${lr_end} \
       --do-train\
       --do-predict\
       --ckpt-frequency=${ckpt_frequency} \
       --weight-decay=${weight_decay} \
       --clip-grad=${clip_grad} \
       --output-dir=${save} \
       --worker-cnt=${worker_cnt} \
       --gpus-per-node=${gpus_per_node} \
