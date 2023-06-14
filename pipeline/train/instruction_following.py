""" Main training script """

import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn
import wandb
from pipeline.train.data import get_data
from pipeline.train.distributed import init_distributed_device, world_info_from_env
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    CLIPImageProcessor,
)

from pipeline.train.train_utils import (
    AverageMeter,
    get_autocast,
    get_cast_dtype,
    get_checkpoint,
)

from flamingo.modeling_flamingo import FlamingoForConditionalGeneration
from flamingo.configuration_flamingo import FlamingoConfig
from otter.modeling_otter import OtterForConditionalGeneration
from otter.configuration_otter import OtterConfig
from tqdm import tqdm
import time

from pipeline.mimicit_utils.arguments import add_data_args
from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights

import sys

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def train_one_epoch(
    args,
    model,
    epoch,
    multi_instruct_loaders,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    accelerator,
    wandb,
):
    num_batches_per_epoch = len(multi_instruct_loaders[0])
    total_training_steps = num_batches_per_epoch * args.num_epochs

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]

    model.train()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter()  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    for num_steps, (batch_multi_instructs) in tqdm(
        enumerate(zip(*multi_instruct_loaders)),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch
        #### MULTI_INSTRUCT FORWARD PASS ####
        total_losses = []
        for batch_multi_instruct in batch_multi_instructs:
            images = batch_multi_instruct["net_input"]["patch_images"]
            input_ids = batch_multi_instruct["net_input"]["input_ids"]
            attention_mask = batch_multi_instruct["net_input"]["attention_masks"]

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[:, 0] = -100

            # remove loss for any token before the first <image> token
            for i in range(labels.shape[0]):
                label_idx = 0
                while label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id:
                    labels[i][label_idx] = -100
                    label_idx += 1

            # # remove loss for any token between <|endofchunk|> and <image>
            # endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            # for endofchunk_idx in endofchunk_idxs:
            #     token_idx = endofchunk_idx + 1
            #     while (
            #         token_idx < labels.shape[1]
            #         and labels[i][token_idx] != media_token_id
            #     ):
            #         labels[i][token_idx] = -100
            #         token_idx += 1

            # <image>User: {cur_incontext_instruction} GPT:<answer> {cur_incontext_answer}<|endofchunk|>User: {instruction} GPT:<answer> {answer}<|endofchunk|>
            # <image>User: {cur_incontext_instruction} GPT:<answer> {cur_incontext_answer}<|endofchunk|><image>User: {instruction} GPT:<answer> {answer}<|endofchunk|>

            # remove loss for any token between first <image> and first <answer>
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            media_idxs = torch.where(labels[i] == media_token_id)[0]
            for media_idx in media_idxs[:1]:
                token_idx = media_idx + 1
                while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
                    labels[i][token_idx] = -100
                    token_idx += 1

            # remove loss for any token between <|endofchunk|> and <answer>, except <image>
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
                    if labels[i][token_idx] == media_token_id:
                        pass
                    else:
                        labels[i][token_idx] = -100
                    token_idx += 1

            labels[labels == answer_token_id] = -100
            labels[labels == media_token_id] = -100

            # import pdb;pdb.set_trace()
            # with accelerator.accumulate(model):
            # with autocast():
            with accelerator.autocast():
                loss_multi_instruct = model(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )[0]
                # loss_multi_instruct = model.generate(
                #     vision_x=images.to(device_id),
                #     lang_x=input_ids.to(device_id),
                #     attention_mask=attention_mask.to(device_id),
                #     max_length=256,
                # )
            total_losses.append(loss_multi_instruct)
        # import pdb;pdb.set_trace()
        #### BACKWARD PASS ####
        total_loss_sum = sum(total_losses)
        mean_loss = total_loss_sum / len(total_losses)
        accelerator.backward(total_loss_sum.to(device_id))

        def mask_embedding(m):
            if m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[answer_token_id] = torch.ones_like(zero_mask[answer_token_id])
                m.weight.grad = m.weight.grad * zero_mask

        if args.mask_lm_head:
            model.module.lang_encoder.model.embed_tokens.apply(mask_embedding)
            model.module.lang_encoder.lm_head.apply(mask_embedding)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # step time and reset end outside of rank 0
        step_time_m.update(time.time() - end)
        end = time.time()

        if accelerator.sync_gradients:
            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                multi_instruct_samples_per_second = args.gradient_accumulation_steps * args.batch_size * args.world_size / step_time_m.val
                multi_instruct_samples_per_second_per_gpu = args.gradient_accumulation_steps * args.batch_size / step_time_m.val

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "multi_instruct_samples_per_second": multi_instruct_samples_per_second,
                        "multi_instruct_samples_per_second_per_gpu": multi_instruct_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_multi_instruct": mean_loss.item(),
                        "global_step": global_step // args.gradient_accumulation_steps,
                    },
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss Multi-Instruct: {mean_loss.item():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help="how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--external_save_dir",
        type=str,
        default=None,
        help="set to save model to external path",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="otter_9b",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100, help="log loss every n steps")
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="path to huggingface model or model identifier from local path or huggingface.co",
        default=None,
    )
    parser.add_argument(
        "--load_from_original_checkpoint",
        type=str,
        help="path to openflamingo provided checkpoint, in .pt format",
        default=None,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite_checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument(
        "--multi_instruct_path",
        type=str,
        help="path to multi_instruct dataset, this should be a glob pattern such as vision_language_examples.tsv",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        help="path to images_path dataset, this should be a glob pattern such as vision_language_examples.tsv",
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        help="path to train_config_path dataset, this should be a glob pattern such as vision_language_examples.tsv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--loss_multiplier_multi_instruct", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--warmup_steps_ratio", default=None, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train_num_samples", type=int, default=None)
    parser.add_argument("--dataset_resampled", action="store_true")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # YH: Training detail
    parser.add_argument("--mask_lm_head", action="store_true")
    # this could potentially save 33GB of all model parameters for otter-9b, including the language and vision model.
    parser.add_argument("--save_hf_model", default=False, action="store_true")
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )

    parser = add_data_args(parser)
    args = parser.parse_args()

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # if args.world_size > 1:
    #     device_id = init_distributed_device(args)
    # else:
    #     device_id = 0
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    device_id = accelerator.device

    random_seed(args.seed)

    if args.pretrained_model_name_or_path is not None:
        if "otter" in args.pretrained_model_name_or_path:
            model = OtterForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                device_map="auto",
                local_files_only=args.offline,
            )
        elif "flamingo" in args.pretrained_model_name_or_path:
            model = FlamingoForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                device_map="auto",
                local_files_only=args.offline,
            )
            model.text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]})
    else:
        config = FlamingoConfig.from_json_file("./flamingo/config.json")
        model = FlamingoForConditionalGeneration(config=config)

        """
        TODO: deprecate this option since the original checkpoints are not supported in future versions
        TODO: all future checkpoints (even released from openflamingo), we will convert them and save to huggingface format.
        TODO: supposedly using "args.pretrained_model_name_or_path" should be the best way to load the model.
        """
        if args.load_from_original_checkpoint is not None:
            print(f"Loading checkpoint from {args.load_from_original_checkpoint}")
            model.load_state_dict(
                torch.load(args.load_from_original_checkpoint, map_location="cpu"),
                False,
            )

    model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))
    args.tokenizer = model.text_tokenizer
    tokenizer = model.text_tokenizer
    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    # device_id = args.rank % torch.cuda.device_count()

    multi_instruct_loaders = get_data(args, tokenizer, "multi_instruct")

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return "gated_cross_attn_layer" in x and "ff_gate" not in x and "attn_gate" not in x and "norm" not in x and "bias" not in x

        for n, p in model.named_parameters():
            # if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    total_training_steps = len(multi_instruct_loaders[0]) * args.num_epochs

    resume_from_epoch = 0
    # check if a checkpoint exists for this run
    args.external_save_dir = os.path.join(args.external_save_dir, args.run_name) if args.external_save_dir else args.run_name
    if os.path.exists(f"{args.external_save_dir}") and args.resume_from_checkpoint is True:
        checkpoint_list = glob.glob(f"{args.external_save_dir}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.external_save_dir}.")
        else:
            resume_from_checkpoint_path = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print(f"Found checkpoint {resume_from_checkpoint_path} for run {args.external_save_dir}.")

        if args.rank == 0:
            print(f"Loading checkpoint from {resume_from_checkpoint_path}")
        checkpoint = torch.load(resume_from_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1

    optimizer = torch.optim.AdamW(get_grouped_params(model), lr=args.learning_rate)

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    args.warmup_steps = total_training_steps * args.warmup_steps_ratio if args.warmup_steps_ratio is not None else args.warmup_steps

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    model, optimizer, lr_scheduler, multi_instruct_loaders = accelerator.prepare(model, optimizer, lr_scheduler, multi_instruct_loaders)
    model.train()

    # device_id = accelerator.device

    for epoch in range(resume_from_epoch, args.num_epochs):
        for cur_data_loader in multi_instruct_loaders:
            cur_data_loader.dataset.set_epoch(epoch)

        train_one_epoch(
            args=args,
            model=model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            multi_instruct_loaders=multi_instruct_loaders,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
        )
        if args.rank == 0:
            if not os.path.exists(args.external_save_dir):
                os.makedirs(args.external_save_dir)

        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if args.rank == 0:
        if not os.path.exists(args.external_save_dir):
            os.makedirs(args.external_save_dir)

        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(
            get_checkpoint(model=unwrapped_model),
            f"{args.external_save_dir}/final_weights.pt",
        )
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.external_save_dir}/final_weights.pt")
        if args.save_hf_model:
            model.save_pretrained(f"{args.external_save_dir}")


if __name__ == "__main__":
    main()
