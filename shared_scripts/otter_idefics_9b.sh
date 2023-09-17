cd /mnt/petrelfs/libo.p/Otter

export PYTHONPATH=.

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_zero3.yaml \
    pipeline/train/instruction_following.py \
    --pretrained_model_name_or_path=/mnt/petrelfs/share_data/zhangyuanhan/otter/otter_idefics9b_instruct_init/ \
    --model_name=idefics \
    --instruction_format=idefics \
    --mimicit_path="/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/release_format/LACR_T2T_instructions.json,/mnt/petrelfs/zhangyuanhan/data/m3it/reasoning/scienceqa/scienceqa_instructions.json,/mnt/petrelfs/zhangyuanhan/data/m3it/captioning/coco/coco_instructions.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/vqav2/vqav2_instructions.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/text-vqa/text-vqa_instructions.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/okvqa/okvqa_instructions.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/a-okvqa/a-okvqa_instructions.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/release_format/LACONV_instructions.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/release_format/LADD_instructions.json" \
    --images_path="/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/release_format/LA.json,/mnt/petrelfs/zhangyuanhan/data/m3it/reasoning/scienceqa/scienceqa.json,/mnt/petrelfs/zhangyuanhan/data/m3it/captioning/coco/coco.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/vqav2/vqav2.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/text-vqa/text-vqa.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/okvqa/okvqa.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/a-okvqa/a-okvqa.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/release_format/LA.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/release_format/LA.json" \
    --train_config_path='/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/release_format/LACR_T2T_train_basic.json,/mnt/petrelfs/zhangyuanhan/data/m3it/reasoning/scienceqa/scienceqa_train.json,/mnt/petrelfs/zhangyuanhan/data/m3it/captioning/coco/coco_train.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/vqav2/vqav2_train.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/text-vqa/text-vqa_train.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/okvqa/okvqa_train.json,/mnt/petrelfs/zhangyuanhan/data/m3it/vqa/a-okvqa/a-okvqa_train.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/release_format/LADD_train.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LA/release_format/LACONV_train.json' \
    --mimicit_text_path="/mnt/petrelfs/zhangyuanhan/data/mimicit/LANG_Only/LIMA/LIMA_instructions.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LANG_Only/AL/AL_instructions.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LANG_Only/CAL/CAL_instructions.json,/mnt/petrelfs/share_data/libo/ORCACHAT/ORCACHAT_instructions.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LANG_Only/MBPP/MBPP_Instruction.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LANG_Only/GUANACO/guanaco_instructions.json" \
    --train_config_text_path="/mnt/petrelfs/zhangyuanhan/data/mimicit/LANG_Only/LIMA/LIMA_train.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LANG_Only/AL/AL_train.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LANG_Only/CAL/CAL_train.json,/mnt/petrelfs/share_data/libo/ORCACHAT/ORCACHAT_train.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LANG_Only/MBPP/MBPP_train.json,/mnt/petrelfs/zhangyuanhan/data/mimicit/LANG_Only/GUANACO/guanaco_train.json" \
    --batch_size=1 \
    --num_epochs=1 \
    --report_to_wandb \
    --wandb_entity=ntu-slab \
    --external_save_dir=/mnt/petrelfs/libo.p/Otter/checkpoints \
    --run_name=otter_idefics9b_0908 \
    --wandb_project=otter_idefics9b \
    --workers=16 \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --save_hf_model \
    --save_ckpt_each_epoch