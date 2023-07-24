""" Main training script """

import argparse
import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import wandb
from flamingo.modeling_flamingo import FlamingoForConditionalGeneration
from otter.modeling_otter import OtterForConditionalGeneration
from pipeline.train.data import get_data
from pipeline.train.distributed import world_info_from_env
from pipeline.train.train_utils import AverageMeter, get_checkpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external_save_dir",
        type=str,
        default=None,
        help="set to save model to external path",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Whether to resume from checkpoint, if set True, will load models from --external_save_dir",
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="otter_9b",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--mmc4_shards",
        type=str,
        help="path to c4 shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument(
        "--laion_shards",
        type=str,
        help="path to laion shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument("--train_num_samples_mmc4", type=int, default=100)
    parser.add_argument("--train_num_samples_laion", type=int, default=100)
    parser.add_argument("--batch_size_mmc4", type=int, default=8)
    parser.add_argument("--batch_size_laion", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset_resampled", action="store_true")
    parser.add_argument(
        "--mmc4_textsim_threshold",
        default=0.32,
        type=float,
        help="threshold for filtering images in mmc4 based on image-text similarity",
    )

    # parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100, help="log loss every n steps")
    parser.add_argument("--checkpointing_steps", type=int, default=10000, help="checkpointing every n steps")
    # Sum of gradient optimization batch size

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="path to huggingface model or model identifier from local path or huggingface.co",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--loss_multiplier_mmc4", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_laion", type=float, default=0.2)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--warmup_steps_ratio", default=None, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
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
    parser.add_argument(
        "--max-src-length",
        type=int,
        default=1024,
        help="the maximum src sequence length",
    )
    parser.add_argument(
        "--max-tgt-length",
        type=int,
        default=1024,
        help="the maximum target sequence length",
    )
    parser.add_argument("--patch-image-size", type=int, default=224)
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
    return parser


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def train_one_epoch(args, model, epoch, mmc4_loader, laion_loader, tokenizer, optimizer, lr_scheduler, device_id, accelerator, wandb):
    num_batches_per_epoch_laion = laion_loader.num_batches
    num_batches_per_epoch_mmc4 = mmc4_loader.num_batches

    assert num_batches_per_epoch_laion == num_batches_per_epoch_mmc4, "Number of batches in laion and mmc4 datasets must be the same"

    num_batches_per_epoch = num_batches_per_epoch_mmc4
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
    for num_steps, (batch_laion, batch_mmc4) in tqdm(
        enumerate(zip(laion_loader, mmc4_loader)),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch
        total_losses = []

        #### LAION FORWARD PASS ####
        images = batch_laion[0].to(device_id, non_blocking=True).unsqueeze(1).unsqueeze(1)

        input_ids = batch_laion[1][0].to(device_id, non_blocking=True)
        attention_mask = batch_laion[1][1].to(device_id, non_blocking=True)

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == media_token_id] = -100
        labels.to(device_id)

        with accelerator.autocast():
            loss_laion = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

        # model.eval()
        # model.text_tokenizer.padding_side = "left"
        # text_prompt_lang_x = model.text_tokenizer(
        #     [
        #         "<image>",
        #     ],
        #     return_tensors="pt",
        # )['input_ids']
        # outputs_debug = model.generate(
        #     vision_x=images.to(device_id),
        #     lang_x=text_prompt_lang_x.to(device_id),
        #     attention_mask=attention_mask.to(device_id),
        #     max_length=256,
        # )

        # print(model.text_tokenizer.batch_decode(outputs_debug))
        # print(model.text_tokenizer.batch_decode(input_ids))
        # model.train()

        #### LAION BACKWARD ####
        accelerator.backward(args.loss_multiplier_laion * loss_laion)
        total_losses.append(args.loss_multiplier_laion * loss_laion)

        #### MMC4 FORWARD PASS ####
        images = batch_mmc4[0].to(device_id, non_blocking=True).unsqueeze(2)
        input_ids = torch.stack([x[0] for x in batch_mmc4[1]]).squeeze(1)
        attention_mask = torch.stack([x[1] for x in batch_mmc4[1]]).squeeze(1)

        # NOTE: irena: expected shape of clip_text_input_ids / attention_mask is (N, I, max_seq_len)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100

        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id:
                labels[i][label_idx] = -100
                label_idx += 1

            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while token_idx < labels.shape[1] and labels[i][token_idx] != media_token_id:
                    labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == media_token_id] = -100
        labels.to(device_id)

        # with accelerator.accumulate(model):
        with accelerator.autocast():
            loss_mmc4 = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

        # model.text_tokenizer.padding_side = "left"
        # outputs_debug = model.generate(
        #     vision_x=images.to(device_id),
        #     lang_x=input_ids.to(device_id),
        #     attention_mask=attention_mask.to(device_id),
        #     max_length=256,
        # )

        # print(model.text_tokenizer.batch_decode(outputs_debug))
        # print(model.text_tokenizer.batch_decode(input_ids))

        #### MMC4 BACKWARD ####
        accelerator.backward(args.loss_multiplier_mmc4 * loss_mmc4)
        total_losses.append(args.loss_multiplier_mmc4 * loss_mmc4)
        #### Collect MMC4/LAION Loss Info ####
        total_loss_sum = sum(total_losses)
        mean_loss = total_loss_sum / len(total_losses)
        # accelerator.backward(total_loss_sum.to(device_id))

        def mask_embedding(m):
            if m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                # zero_mask[answer_token_id] = torch.ones_like(zero_mask[answer_token_id])
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
                m.weight.grad = m.weight.grad * zero_mask

        if args.mask_lm_head:
            unwrapped_model = accelerator.unwrap_model(model)
            if unwrapped_model.lang_encoder.__class__.__name__ == "MPTForCausalLM":
                unwrapped_model.lang_encoder.transformer.wte.apply(mask_embedding)
            elif unwrapped_model.lang_encoder.__class__.__name__ == "LlamaForCausalLM":
                unwrapped_model.lang_encoder.model.embed_tokens.apply(mask_embedding)
                unwrapped_model.lang_encoder.lm_head.apply(mask_embedding)

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
                mmc4_samples_per_second = args.gradient_accumulation_steps * args.batch_size_mmc4 * args.world_size / step_time_m.val
                mmc4_samples_per_second_per_gpu = args.gradient_accumulation_steps * args.batch_size_mmc4 / step_time_m.val
                laion_samples_per_second = args.gradient_accumulation_steps * args.batch_size_laion * args.world_size / step_time_m.val
                laion_samples_per_second_per_gpu = args.gradient_accumulation_steps * args.batch_size_laion / step_time_m.val
                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "mmc4_samples_per_second": mmc4_samples_per_second,
                        "mmc4_samples_per_second_per_gpu": mmc4_samples_per_second_per_gpu,
                        "laion_samples_per_second": laion_samples_per_second,
                        "laion_samples_per_second_per_gpu": laion_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "mmc4_loss": loss_mmc4.item(),
                        "laion_loss": loss_laion.item(),
                        "mean_loss": mean_loss.item(),
                        "global_step": global_step // args.gradient_accumulation_steps,
                    },
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Mean Loss: {mean_loss.item():.3f}")
        # Add a process on saving checkpoints during pretraining
        if ((num_steps + 1) % args.checkpointing_steps == 0) and args.rank == 0:
            if not os.path.exists(args.external_save_dir):
                os.makedirs(args.external_save_dir)

            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(unwrapped_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_steps{num_steps + 1}.pt")
            accelerator.save(checkpoint_dict, f"{args.external_save_dir}/checkpoint_steps{num_steps + 1}.pt")
            # save the config
            print(f"Saving config to {args.external_save_dir}/config.json")
            unwrapped_model.config.save_pretrained(args.external_save_dir)
            if args.delete_previous_checkpoint:
                if (num_steps + 1) // args.checkpointing_steps >= 2:
                    previous_checkpoint_path = f"{args.external_save_dir}/checkpoint_steps{num_steps + 1 - args.checkpointing_steps}.pt"
                    if os.path.exists(previous_checkpoint_path):
                        os.remove(previous_checkpoint_path)


def main():
    parser = parse_args()
    # TODO: remove additional data args, all args would be processed in above parser
    # parser = add_data_args(parser)
    args = parser.parse_args()

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    device_id = accelerator.device

    random_seed(args.seed)
    if args.pretrained_model_name_or_path is not None:
        accelerator.print(f"Loading pretrained model from {args.pretrained_model_name_or_path}")
        if "otter" in args.run_name.lower():
            model = OtterForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                device_map="auto",
                local_files_only=args.offline,
            )
        elif "flamingo" in args.run_name.lower():
            if accelerator.num_processes > 1:
                model = FlamingoForConditionalGeneration.from_pretrained(
                    args.pretrained_model_name_or_path,
                    device_map={"": device_id},
                    local_files_only=args.offline,
                )
            else:
                model = FlamingoForConditionalGeneration.from_pretrained(
                    args.pretrained_model_name_or_path,
                    device_map="auto",
                    local_files_only=args.offline,
                )
            model.text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]})
    else:
        model = None

    accelerator.wait_for_everyone()

    if model.lang_encoder.__class__.__name__ != "MPTForCausalLM":
        model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))

    args.tokenizer = model.text_tokenizer
    tokenizer = model.text_tokenizer
    random_seed(args.seed, args.rank)

    image_processor = CLIPImageProcessor()

    mmc4_dataset = get_data(args, image_processor, tokenizer, "mmc4")
    laion_dataset = get_data(args, image_processor, tokenizer, "laion")

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

    # total_training_steps = ((args.train_num_samples_mmc4) // (args.batch_size_mmc4 * args.world_size)) * args.num_epochs
    total_training_steps = mmc4_dataset.dataloader.num_batches * args.num_epochs

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

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # YH: hardcode for ddp, reason is related to "split_batch" in accelerator. Currently just fix this bug, need to dig further.
    if accelerator.num_processes > 1:
        lr_scheduler.split_batches = True

    model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        laion_dataset.set_epoch(epoch)
        laion_loader = laion_dataset.dataloader

        mmc4_dataset.set_epoch(epoch)
        mmc4_loader = mmc4_dataset.dataloader

        train_one_epoch(
            args=args,
            model=model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            mmc4_loader=mmc4_loader,
            laion_loader=laion_loader,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
        )
        if args.rank == 0:
            if not os.path.exists(args.external_save_dir):
                os.makedirs(args.external_save_dir)

            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(unwrapped_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_epoch{epoch}.pt")
            accelerator.save(checkpoint_dict, f"{args.external_save_dir}/checkpoint_epoch{epoch}.pt")
            # save the config
            unwrapped_model.config.save_pretrained(args.external_save_dir)
            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(f"{args.external_save_dir}/checkpoint_epoch{epoch-1}.pt")

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
        # save the config
        unwrapped_model.config.save_pretrained(args.external_save_dir)

        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.external_save_dir}/final_weights.pt")
        if args.save_hf_model:
            unwrapped_model.save_pretrained(f"{args.external_save_dir}")


if __name__ == "__main__":
    main()
