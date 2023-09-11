export PYTHONPATH=.
cd /home/luodian/projects/Otter

python -m accelerate.commands.launch \
--config_file=pipeline/accelerate_configs/accelerate_config_zero3.yaml \
--num_processes=2 \
--main_process_port=12355 \
pipeline/train/instruction_following.py \
--pretrained_model_name_or_path=/home/luodian/projects/checkpoints/otter_idefics9b_0830 \
--mimicit_ic_path=/home/luodian/projects/data/LA/LACR_T2T_instructions.json \
--images_ic_path=/home/luodian/projects/data/LA/LA.json \
--train_config_ic_path=/home/luodian/projects/data/LA/LACR_T2T_train.json --mimicit_path=/home/luodian/projects/data/LA/LACONV_instructions.json --images_path=/home/luodian/projects/data/LA/LA.json --train_config_path=/home/luodian/projects/data/LA/LACONV_train.json --mimicit_text_path=/home/luodian/projects/data/LANG_Only/ORCACHAT/ORCACHAT_instructions.json --train_config_text_path=/home/luodian/projects/data/LANG_Only/ORCACHAT/ORCACHAT_train.json --model_name=idefics --batch_size=1 --num_epochs=1 --report_to_wandb --wandb_entity=ntu-slab --external_save_dir=/home/luodian/projects/checkpoints --save_hf_model --run_name=Otter-Idefics-Mix --wandb_project=Otter-Various-Instruction --workers=4 --lr_scheduler=cosine --learning_rate=1e-5 --warmup_steps_ratio=0.01 --inst_format=idefics --max_seq_len=2048
