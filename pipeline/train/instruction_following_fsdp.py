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
)
from pipeline.src.collie_lm import FlamingoLMMixin
from open_flamingo import create_model_and_transforms
from open_flamingo.train.train_utils import (
    AverageMeter,
    get_autocast,
    get_cast_dtype,
    get_checkpoint,
)
from tqdm import tqdm
import time

from ofa_compress.arguments import add_data_args
from accelerate import Accelerator
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def save_fsdp(
    sharded_model,
    epoch=0,
    optimizer=None,
    lr_scheduler=None,
    final_weight=False,
    args=None,
):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        sharded_model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        cpu_state = get_checkpoint(sharded_model)

    checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": cpu_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }
    if final_weight:
        torch.save(checkpoint_dict, f"{args.external_save_dir}/final_weight.pt")
    else:
        torch.save(checkpoint_dict, f"{args.external_save_dir}/checkpoint_{epoch}.pt")


def train_one_epoch(
    args,
    model,
    epoch,
    multi_instruct_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    accelerator,
    wandb,
):
    # num_batches_per_epoch = multi_instruct_loader.num_batches
    num_batches_per_epoch = len(multi_instruct_loader)
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    for num_steps, (batch_multi_instruct) in tqdm(
        enumerate(multi_instruct_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch
        #### MULTI_INSTRUCT FORWARD PASS ####

        images = (
            batch_multi_instruct["net_input"]["patch_images"]
            .to(device_id, dtype=cast_dtype, non_blocking=True)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        input_ids = batch_multi_instruct["net_input"]["input_ids"].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )
        attention_mask = batch_multi_instruct["net_input"]["attention_masks"].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100

        for i in range(labels.shape[0]):
            # remove loss for any token before <answer> token
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != answer_token_id
            ):
                labels[i][label_idx] = -100
                label_idx += 1

        labels[labels == answer_token_id] = -100
        labels[labels == media_token_id] = -100

        labels.to(device_id, dtype=cast_dtype, non_blocking=True)

        with autocast():
            loss_multi_instruct = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

        divided_loss_multi_instruct = (
            loss_multi_instruct / args.gradient_accumulation_steps
        )

        #### BACKWARD PASS ####
        accelerator.backward(divided_loss_multi_instruct)

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                zero_mask[answer_token_id] = torch.ones_like(zero_mask[answer_token_id])
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                multi_instruct_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                multi_instruct_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps * args.batch_size / step_time_m.val
                )

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
                        "loss_multi_instruct": divided_loss_multi_instruct.item(),
                        "global_step": global_step,
                    },
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss Multi-Instruct: {loss_multi_instruct.item():.3f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
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
        default="openflamingo3B",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
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
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train_num_samples", type=int, default=10000)
    # parser.add_argument("--train_num_samples_laion", type=int, default=10000)
    parser.add_argument("--dataset_resampled", action="store_true")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
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

    if args.world_size > 1:
        device_id = init_distributed_device(args)
    else:
        device_id = 0

    random_seed(args.seed)

    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        use_media_placement_augmentation=args.use_media_placement_augmentation,
    )

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    # check if a checkpoint exists for this run
    args.external_save_dir = (
        os.path.join(args.external_save_dir, args.run_name)
        if args.external_save_dir
        else args.run_name
    )
    if (
        os.path.exists(f"{args.external_save_dir}")
        and args.resume_from_checkpoint is None
        and args.overwrite_checkpoint is False
    ):
        checkpoint_list = glob.glob(f"{args.external_save_dir}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.external_save_dir}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {args.resume_from_checkpoint} for run {args.external_save_dir}."
            )

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint, False)

    accelerator = Accelerator()
    # model = accelerator.prepare(model)
    my_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            FlamingoLMMixin,
        },
    )
    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )
    torch.cuda.set_device(device_id)
    model = FSDP(
        model,
        auto_wrap_policy=my_wrap_policy,
        mixed_precision=bfSixteen,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )  # Zero-3: params, grads, optimizer states
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    device_id = args.rank % torch.cuda.device_count()

    multi_instruct_dataset = get_data(
        args, image_processor, tokenizer, "multi_instruct"
    )

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                "gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

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

    optimizer = torch.optim.AdamW(get_grouped_params(model), lr=args.learning_rate)

    args.train_num_samples = multi_instruct_dataset.dataloader.num_samples
    total_training_steps = (
        (args.train_num_samples) // (args.batch_size * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    multi_instruct_loader = multi_instruct_dataset.dataloader
    optimizer, multi_instruct_loader = accelerator.prepare(
        optimizer, multi_instruct_loader
    )
    model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        multi_instruct_dataset.set_epoch(epoch)

        train_one_epoch(
            args=args,
            model=model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            multi_instruct_loader=multi_instruct_loader,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
        )
        if args.rank == 0:
            if not os.path.exists(args.external_save_dir):
                os.makedirs(args.external_save_dir)

            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(
                {
                    "model": get_checkpoint(model=unwrapped_model),
                    "optimizer": optimizer.optimizer.state_dict(),  # optimizer is an AcceleratedOptimizer object
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                f"{args.external_save_dir}/checkpoint_{epoch}.pt",
            )
            # print(f"save model at ./bundle.pth")
            if args.report_to_wandb and args.save_checkpoints_to_wandb:
                wandb.save(f"{args.external_save_dir}/checkpoint_{epoch}.pt")

            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(f"{args.external_save_dir}/checkpoint_{epoch-1}.pt")

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


if __name__ == "__main__":
    main()
