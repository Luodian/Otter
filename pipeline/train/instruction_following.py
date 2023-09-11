""" Main training script """

import argparse
import glob
import os
import random
import time

import numpy as np
import gc
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

from otter_ai import OtterForConditionalGeneration
from otter_ai import FlamingoForConditionalGeneration
from pipeline.train.data import get_data
from pipeline.train.distributed import world_info_from_env
from pipeline.train.train_utils import AverageMeter, get_checkpoint, get_image_attention_mask
from transformers import AutoProcessor

import deepspeed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Try importing IdeficsForVisionText2Text, and if it's not available, define a dummy class
try:
    from transformers import IdeficsForVisionText2Text
except ImportError:
    print("IdeficsForVisionText2Text does not exist")
    IdeficsForVisionText2Text = type(None)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def train_one_epoch(args, model, epoch, mimicit_loaders, tokenizer, optimizer, lr_scheduler, device_id, accelerator, wandb):
    num_batches_per_epoch = len(mimicit_loaders[0])
    total_training_steps = num_batches_per_epoch * args.num_epochs

    # special design for Idefics Model's prompt strategy
    fake_token_image_exists = True if "<fake_token_around_image>" in tokenizer.special_tokens_map["additional_special_tokens"] else False
    fake_token_image_token_id = tokenizer("<fake_token_around_image>", add_special_tokens=False)["input_ids"][-1]

    # normal prompt strategy
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_text = (
        "<|endofchunk|>" if "<|endofchunk|>" in tokenizer.special_tokens_map["additional_special_tokens"] else "<end_of_utterance>"
    )  # for different tokenizer
    endofchunk_token_id = tokenizer(endofchunk_text, add_special_tokens=False)["input_ids"][-1]
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
    ens_token_id = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"][-1]

    model.train()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter()  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()
    autocast_type = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32

    # loop through dataloader
    for num_steps, (batch_mimicits) in tqdm(
        enumerate(zip(*mimicit_loaders)),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch
        #### MIMIC-IT FORWARD PASS ####

        total_losses = []
        for batch_mimicit in batch_mimicits:
            images = batch_mimicit["net_input"]["patch_images"].to(device_id, non_blocking=True)
            input_ids = batch_mimicit["net_input"]["input_ids"].to(device_id, non_blocking=True)
            attention_mask = batch_mimicit["net_input"]["attention_masks"].to(device_id, non_blocking=True)

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[:, 0] = -100
            for i in range(labels.shape[0]):
                # get index of all endofchunk/media tokens in the sequence
                endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
                media_idxs = torch.where(labels[i] == media_token_id)[0]

                # remove loss for any token the before the first <answer>
                token_idx = 0
                while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
                    labels[i][token_idx] = -100
                    token_idx += 1

                # remove loss for any token between <|endofchunk|> and <answer>, except <image>
                for endofchunk_idx in endofchunk_idxs[:-1]:
                    token_idx = endofchunk_idx + 1
                    while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
                        if labels[i][token_idx] == media_token_id:
                            pass
                        else:
                            labels[i][token_idx] = -100
                        token_idx += 1

            labels[labels == answer_token_id] = -100
            labels[labels == media_token_id] = -100
            if fake_token_image_exists:
                labels[labels == fake_token_image_token_id] = -100

            with accelerator.autocast():
                unwrapped_model = accelerator.unwrap_model(model)
                if num_steps == 0:
                    # info check
                    accelerator.print(f"input_ids: {input_ids.shape}")
                    accelerator.print(f"images: {images.shape}")
                    accelerator.print(f"attention_mask: {attention_mask.shape}")
                    accelerator.print(f"labels: {labels.shape}")
                    accelerator.print(f"model: {unwrapped_model.__class__.__name__}")
                    accelerator.print(f"model dtype: {unwrapped_model.dtype}")

                if IdeficsForVisionText2Text is not None and isinstance(unwrapped_model, IdeficsForVisionText2Text):
                    # only for image model
                    max_num_images = images.shape[1]
                    pure_text = torch.all(images == 0)
                    image_attention_mask = get_image_attention_mask(input_ids, max_num_images, tokenizer, include_image=not pure_text)
                    # assert images.shape[1] == 1, "The second dimension is not 1"

                    loss_mimicit = model(
                        pixel_values=images.squeeze(1).to(autocast_type),
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_attention_mask=image_attention_mask,
                        labels=labels,
                    )[0]
                else:
                    loss_mimicit = model(
                        vision_x=images.to(autocast_type),
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )[0]

            if accelerator.mixed_precision == "fp16":
                accelerator.backward(loss_mimicit.to(device_id))
            else:
                accelerator.backward(loss_mimicit)

            total_losses.append(loss_mimicit)
        #### BACKWARD PASS ####
        total_loss_sum = sum(total_losses)
        mean_loss = total_loss_sum / len(total_losses)
        # accelerator.backward(total_loss_sum.to(device_id))

        def mask_embedding(m):
            if m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[answer_token_id] = torch.ones_like(zero_mask[answer_token_id])
                # zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                # zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
                m.weight.grad = m.weight.grad * zero_mask

        if args.mask_lm_head and args.distributed_type != "DEEPSPEED":
            unwrapped_model = accelerator.unwrap_model(model)
            if isinstance(unwrapped_model, IdeficsForVisionText2Text):
                # This code need to be refined.
                unwrapped_model.lm_head.apply(mask_embedding)
            elif unwrapped_model.lang_encoder.__class__.__name__ in ["MPTForCausalLM", "MosaicGPT"]:
                unwrapped_model.lang_encoder.transformer.wte.apply(mask_embedding)
            elif "LlamaForCausalLM" in unwrapped_model.lang_encoder.__class__.__name__:
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
                mimicit_samples_per_second = args.gradient_accumulation_steps * args.batch_size * args.world_size / step_time_m.val
                mimicit_samples_per_second_per_gpu = args.gradient_accumulation_steps * args.batch_size / step_time_m.val

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "mimicit_samples_per_second": mimicit_samples_per_second,
                        "mimicit_samples_per_second_per_gpu": mimicit_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_mimicit": mean_loss.item(),
                        "global_step": global_step // args.gradient_accumulation_steps,
                    },
                    commit=True,
                )
                # torch.cuda.empty_cache()
                # gc.collect()  # forces garbage collection

            if args.rank == 0 and global_step != 0 and (args.save_steps_interval != -1) and (global_step % args.save_steps_interval == 0):
                if not os.path.exists(args.external_save_dir):
                    os.makedirs(args.external_save_dir)

                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_dict = {
                    "steps": global_step,
                    "model_state_dict": get_checkpoint(unwrapped_model),
                }
                print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_steps_{global_step}.pt")
                accelerator.save(checkpoint_dict, f"{args.external_save_dir}/checkpoint_steps_{global_step}.pt")
                if args.delete_previous_checkpoint:
                    if epoch > 0 and os.path.exists(f"{args.external_save_dir}/checkpoint_step_{global_step-args.save_steps_interval}.pt"):
                        os.remove(f"{args.external_save_dir}/checkpoint_step_{global_step-args.save_steps_interval}.pt")

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss MIMIC-IT: {mean_loss.item():.3f}")


def parse_args():
    """
    Parse the command line arguments and perform the initial setup.
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Main training script for the model")

    # Add arguments to the parser
    # TODO: Add help messages to clarify the purpose of each argument

    # Model configuration arguments
    parser.add_argument(
        "--external_save_dir",
        type=str,
        default=None,
        help="set to save model to external path",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="otter-9b",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="otter",
        choices=["otter", "flamingo", "idefics"],
        help="otters or flamingo",
    )
    parser.add_argument(
        "--inst_format",
        type=str,
        default="simple",
        choices=["simple", "llama2", "idefics"],
        help="simple is for mpt/llama1, rest are in different instruction templates.",
    )
    # Prepare the arguments for different types of data sources.
    # Arguments are grouped by data types and whether the data is from past or new sources.
    # Arguments for image-text data, including multi-run conversations.
    parser.add_argument(
        "--past_mimicit_path",
        type=str,
        default="",
        help="Path to the past image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--past_images_path",
        type=str,
        default="",
        help="Path to the past images dataset (including base64 format images). Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--past_train_config_path",
        type=str,
        default="",
        help="Path to the past images dataset (including current ids and related in-context ids). Should be in format /path/to/xx_train.json",
    )

    parser.add_argument(
        "--mimicit_path",
        type=str,
        default="",
        help="Path to the new image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="",
        help="Path to the new images dataset (including base64 format images). Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        default="",
        help="Path to the new images dataset (including current ids and related in-context ids). Should be in format /path/to/xx_train.json",
    )

    # Arguments for image-text in-context data.
    parser.add_argument(
        "--past_mimicit_ic_path",
        type=str,
        default="",
        help="Path to the past in-context image-text dataset. Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--past_images_ic_path",
        type=str,
        default="",
        help="Path to the past in-context images dataset. Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--past_train_config_ic_path",
        type=str,
        default="",
        help="Path to the past in-context training config dataset. Should be in format /path/to/xx_train.json",
    )
    parser.add_argument(
        "--mimicit_ic_path",
        type=str,
        default="",
        help="Path to the new in-context image-text dataset. Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--images_ic_path",
        type=str,
        default="",
        help="Path to the new in-context images dataset. Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--train_config_ic_path",
        type=str,
        default="",
        help="Path to the new in-context training config dataset. Should be in format /path/to/xx_train.json",
    )

    # Arguments for text data, including multi-run conversations.
    parser.add_argument(
        "--mimicit_text_path",
        type=str,
        default="",
        help="Path to the new text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--train_config_text_path",
        type=str,
        default="",
        help="Path to the new text dataset (including multi-run conversations). Should be in format /path/to/xx_train.json",
    )
    parser.add_argument(
        "--past_mimicit_text_path",
        type=str,
        default="",
        help="Path to the past text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--past_train_config_text_path",
        type=str,
        default="",
        help="Path to the past text dataset (including multi-run conversations). Should be in format /path/to/xx_train.json",
    )

    # Arguments for video-text data.
    parser.add_argument(
        "--training_data_yaml",
        type=str,
        default="",
        help="Path to the training data yaml file.",
    )
    parser.add_argument(
        "--past_mimicit_vt_path",
        type=str,
        default="",
        help="Path to the past video-text dataset. Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--past_images_vt_path",
        type=str,
        default="",
        help="Path to the past images dataset (associated with video-text data). Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--mimicit_vt_path",
        type=str,
        default="",
        help="Path to the new video-text dataset. Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--images_vt_path",
        type=str,
        default="",
        help="Path to the new images dataset (associated with video-text data). Should be in format /path/to/xx.json",
    )

    # Argument for specifying the ratio for resampling past datasets.
    parser.add_argument(
        "--past_subset_ration",
        type=float,
        default=1.0,
        help="The ratio for resampling the past dataset. Should be a float between 0 and 1.",
    )

    # optimizer args
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--save_ckpt_each_epoch", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100, help="log loss every n steps")
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_num_samples", type=int, default=-1)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps_interval", type=int, default=-1)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="path to huggingface model or model identifier from local path or huggingface.co",
        default=None,
    )
    parser.add_argument(
        "--trained_ckpt",
        type=str,
        help="path to trained_ckpt",
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
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--warmup_steps_ratio", default=None, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--workers", type=int, default=4)
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
        "--max_seq_len",
        type=int,
        default=2048,
        help="the maximum src sequence length",
    )
    parser.add_argument("--patch-image-size", type=int, default=224)
    parser.add_argument("--resample_frames", type=int, default=32)
    # this could potentially save 33GB of all model parameters for otter-9b, including the language and vision model.
    parser.add_argument("--save_hf_model", default=False, action="store_true")
    parser.add_argument(
        "--customized_config",
        default=None,
        type=str,
        help="path to customized additional config.json, use to modify from the original config.json in pretrained model.",
    )
    parser.add_argument("--task_name", default="", type=str, help="task name, used to decide different function to load dataset.")
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
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        action="store_true",
        help="resume from checkpoint (original openflamingo pt format, not hf format)",
    )
    # TODO: remove additional data args, all args would be processed in above parser
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    # parser = add_data_args(parser)
    args = parser.parse_args()

    # Check for argument consistency and set environment variables if needed
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # if "COUNT_NODE" in os.environ:
    #     args.num_machines = int(os.environ["COUNT_NODE"])
    # else:
    #     args.num_machines = 1

    # if "THEID" in os.environ:
    #     args.machine_rank = int(os.environ["THEID"])
    # else:
    #     args.machine_rank = 0

    # Seed for reproducibility
    random_seed(args.seed)

    return args


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="bf16")
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size

    device_id = accelerator.device

    if args.pretrained_model_name_or_path is not None:
        accelerator.print(f"Loading pretrained model from {args.pretrained_model_name_or_path}")
        device_map = {"": device_id} if accelerator.distributed_type == "MULTI_GPU" or accelerator.distributed_type == "DEEPSPEED" else "auto"
        kwargs = {"local_files_only": args.offline, "device_map": device_map}
        if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
            kwargs.pop("device_map")
        if args.customized_config is not None:
            kwargs["config"] = args.customized_config
        if "otter" in args.model_name.lower():
            model = OtterForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                **kwargs,
            )
            args.tokenizer = model.text_tokenizer
            tokenizer = model.text_tokenizer
            image_processor = CLIPImageProcessor()
        elif "flamingo" in args.model_name.lower():
            model = FlamingoForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                **kwargs,
            )
            # add special tokens for instruction tuning
            model.text_tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>"]})
            args.tokenizer = model.text_tokenizer
            tokenizer = model.text_tokenizer
            image_processor = CLIPImageProcessor()
        elif "idefics" in args.model_name.lower():
            # import pdb;pdb.set_trace()
            model = IdeficsForVisionText2Text.from_pretrained(
                args.pretrained_model_name_or_path,
                **kwargs,
            )
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()

            # named_parameters = dict(model.named_parameters())
            # params_to_gather = [named_parameters[k] for k in named_parameters.keys()]
            # if len(params_to_gather) > 0:
            if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
                params_to_gather = [p for name, p in model.named_parameters() if p.requires_grad]
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        # 有参数
                        print(
                            device_id,
                            f"IDEFICS Trainable Params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e9:.3f} B",
                        )
            else:
                print(
                    device_id,
                    f"IDEFICS Trainable Params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e9:.3f} B",
                )
            processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path, legacy=False)
            past_special_tokens = processor.tokenizer.special_tokens_map["additional_special_tokens"]
            processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>"] + past_special_tokens})
            image_processor = processor.image_processor
            tokenizer = processor.tokenizer
            # make embedding size divisible by 64 for hardware compatiblity https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            new_embedding_size = (len(tokenizer) // 64 + 1) * 64
            model.resize_token_embeddings(new_embedding_size, pad_to_multiple_of=64)

    if args.trained_ckpt is not None:
        train_ckpt = torch.load(args.trained_ckpt, map_location="cpu")
        if train_ckpt.get("model_state_dict", None) is not None:
            train_ckpt = train_ckpt["model_state_dict"]
        _ = model.load_state_dict(train_ckpt, strict=False)
        print(_[1])

    accelerator.wait_for_everyone()

    args.distributed_type = accelerator.distributed_type

    if hasattr(model, "lang_encoder") and "LlamaForCausalLM" in model.lang_encoder.__class__.__name__:
        model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    mimicit_loaders = get_data(args, image_processor, tokenizer, "mimicit")

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

    total_training_steps = len(mimicit_loaders[0]) * args.num_epochs

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

    args.warmup_steps = total_training_steps * args.warmup_steps_ratio if args.warmup_steps_ratio is not None else args.warmup_stepsps

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

    if accelerator.distributed_type == "DEEPSPEED" or accelerator.distributed_type == "MULTI_GPU":
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model, optimizer, lr_scheduler, mimicit_loaders = accelerator.prepare(model, optimizer, lr_scheduler, mimicit_loaders)

    model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        for cur_data_loader in mimicit_loaders:
            cur_data_loader.dataset.set_epoch(epoch)

        train_one_epoch(
            args=args,
            model=model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            mimicit_loaders=mimicit_loaders,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
        )
        accelerator.wait_for_everyone()

        if args.save_ckpt_each_epoch:
            if args.rank == 0:
                if not os.path.exists(args.external_save_dir):
                    os.makedirs(args.external_save_dir)

            if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
                checkpoint_dict = accelerator.get_state_dict(model)

                if args.rank == 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    trainable_params_name = [name for name, p in unwrapped_model.named_parameters() if p.requires_grad]
                    for name in list(checkpoint_dict.keys()):
                        if name not in trainable_params_name:
                            del checkpoint_dict[name]

            else:
                if args.rank == 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    # checkpoint_dict = {
                    #     "epoch": epoch,
                    #     "model_state_dict": get_checkpoint(unwrapped_model),
                    #     "optimizer_state_dict": optimizer.state_dict(),
                    #     "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    # }
                    checkpoint_dict = {
                        "model_state_dict": get_checkpoint(unwrapped_model),
                    }

            if args.rank == 0:
                print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_{epoch}.pt")
                accelerator.save(checkpoint_dict, f"{args.external_save_dir}/checkpoint_{epoch}.pt")
                # save the config
                unwrapped_model.config.save_pretrained(args.external_save_dir)
                if args.delete_previous_checkpoint:
                    if epoch > 0:
                        os.remove(f"{args.external_save_dir}/checkpoint_{epoch-1}.pt")

            accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()

    if args.rank == 0:
        if not os.path.exists(args.external_save_dir):
            os.makedirs(args.external_save_dir)

    if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
        checkpoint_dict = accelerator.get_state_dict(model)

        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.config.save_pretrained(args.external_save_dir)

        if args.rank == 0 and not args.save_hf_model:
            trainable_params_name = [name for name, p in unwrapped_model.named_parameters() if p.requires_grad]
            for name in list(checkpoint_dict.keys()):
                if name not in trainable_params_name:
                    del checkpoint_dict[name]

            accelerator.save(
                checkpoint_dict,
                f"{args.external_save_dir}/final_weights.pt",
            )
        elif args.rank == 0 and args.save_hf_model:
            unwrapped_model.save_pretrained(
                f"{args.external_save_dir}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=checkpoint_dict,
            )

    else:
        if args.rank == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_dict = get_checkpoint(model=unwrapped_model)

            accelerator.save(
                checkpoint_dict,
                f"{args.external_save_dir}/final_weights.pt",
            )
            # save the config
            unwrapped_model.config.save_pretrained(args.external_save_dir)

            if args.report_to_wandb and args.save_checkpoints_to_wandb:
                wandb.save(f"{args.external_save_dir}/final_weights.pt")
            if args.save_hf_model:
                unwrapped_model.save_pretrained(f"{args.external_save_dir}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
