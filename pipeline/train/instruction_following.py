""" Main training script """

import argparse
import gc
import glob
import os
import sys
import time
from itertools import cycle

import deepspeed
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import wandb

sys.path.append("../..")
from transformers import AutoProcessor, AutoTokenizer, FuyuImageProcessor
from src.otter_ai.models.fuyu.modeling_fuyu import FuyuForCausalLM
from src.otter_ai.models.fuyu.processing_fuyu import FuyuProcessor

from pipeline.mimicit_utils.data import get_data
from pipeline.train.train_args import parse_args
from pipeline.train.train_utils import (
    AverageMeter,
    get_grouped_params,
    get_image_attention_mask,
    master_print,
    random_seed,
    save_checkpoint,
    save_final_weights,
    verify_yaml,
    get_weights_for_dataloaders,
    get_next_dataloader,
    find_and_remove_tokens,
    delete_tensors_from_dict,
)
from src.otter_ai.models.flamingo.modeling_flamingo import FlamingoForConditionalGeneration
from src.otter_ai.models.otter.modeling_otter import OtterForConditionalGeneration
from transformers import LlamaForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
# Try importing IdeficsForVisionText2Text, and if it's not available, define a dummy class
try:
    from transformers import IdeficsForVisionText2Text
except ImportError:
    print("IdeficsForVisionText2Text does not exist")
    IdeficsForVisionText2Text = type(None)

# from memory_profiler import profile
# fp = open("memory_report.log", "w+")


# @profile(stream=fp)
def forward_pass(args, model, tokenizer, images, input_ids, attention_mask, labels, device_id, autocast_type, batch_mimicit):
    if args.model_name == "fuyu":
        model_inputs = batch_mimicit.pop("fuyu_data")
        for k, v in model_inputs.items():
            model_inputs[k] = v.to(device_id, non_blocking=True) if isinstance(v, torch.Tensor) else [vv.to(device_id, non_blocking=True) for vv in v]
        loss_mimicit = model(**model_inputs)[0]
    elif args.model_name == "idefics":
        # only for image model
        max_num_images = images.shape[1]
        pure_text = torch.all(images == 0)
        image_attention_mask = get_image_attention_mask(
            input_ids,
            max_num_images,
            tokenizer,
            include_image=not pure_text,
        )
        image_attention_mask = image_attention_mask.to(device_id, non_blocking=True)
        loss_mimicit = model(
            pixel_values=images.squeeze(2).to(autocast_type),
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_attention_mask=image_attention_mask,
            labels=labels,
        )[0]
    elif args.model_name == "otter" or args.model_name == "flamingo":
        loss_mimicit = model(
            vision_x=images.to(autocast_type),
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )[0]
    elif args.model_name == "llama2":
        loss_mimicit = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )[0]
    else:
        raise NotImplementedError(f"Loss of model {args.model_name} not implemented.")

    return loss_mimicit


def train_one_epoch(args, model, epoch, mimicit_loaders, tokenizer, optimizer, lr_scheduler, device_id, accelerator, wandb):
    dataloader_iterators = [cycle(dataloader) for dataloader in mimicit_loaders]
    weights = get_weights_for_dataloaders(mimicit_loaders)
    num_batches_per_epoch = sum(len(dataloader) for dataloader in mimicit_loaders) // args.gradient_accumulation_steps

    # Special Design for Idefics Model's prompt strategy
    if args.model_name.lower() == "idefics":
        fake_token_image_exists = True if "<fake_token_around_image>" in tokenizer.special_tokens_map["additional_special_tokens"] else False
        fake_token_image_token_id = tokenizer("<fake_token_around_image>", add_special_tokens=False)["input_ids"][-1]
        endofchunk_text = "<end_of_utterance>"
    else:
        fake_token_image_exists = False
        fake_token_image_token_id = None
        endofchunk_text = "<|endofchunk|>"

    # Normal Prompt Strategy
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer(endofchunk_text, add_special_tokens=False)["input_ids"][-1]
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
    eos_token_id = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"][-1]

    model.train()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter()  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()
    autocast_type = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32

    # loop through different groups of dataloader
    for num_steps in tqdm(range(args.total_training_steps), disable=args.rank != 0, initial=(epoch * num_batches_per_epoch)):
        if num_steps == num_batches_per_epoch:
            break
        data_time_m.update(time.time() - end)
        dataloader_iterator = get_next_dataloader(dataloader_iterators, weights)
        batch_mimicit = next(dataloader_iterator)  # Fetch a batch from the chosen dataloader
        global_step = num_steps + epoch * num_batches_per_epoch

        #### MIMIC-IT FORWARD PASS ####
        net_input = batch_mimicit.pop("net_input")
        images = net_input.pop("patch_images").to(device_id, non_blocking=True)
        input_ids = net_input.pop("input_ids").to(device_id, non_blocking=True)
        attention_mask = net_input.pop("attention_masks").to(device_id, non_blocking=True)
        labels = None  # placeholder to avoid error

        if args.model_name != "fuyu":  # design fuyu's process into it's processor, a way better design than following code.

            def masking(masking_number: int = -100):
                labels = torch.empty(input_ids.shape, dtype=torch.int64).to(device_id, non_blocking=True)
                for i in range(input_ids.shape[0]):
                    labels[i] = torch.where(input_ids[i] == eos_token_id, eos_token_id, masking_number)
                    answer_token_ids_all = torch.where(input_ids[i] == answer_token_id)[0]
                    endofchunk_token_ids_all = torch.where(input_ids[i] == endofchunk_token_id)[0]

                    j = 0  # Counter for endofchunk_token_ids
                    for answer_token_idx in answer_token_ids_all:
                        # Find the closest endofchunk_token_id that is greater than answer_token_id
                        while j < len(endofchunk_token_ids_all) and endofchunk_token_ids_all[j] < answer_token_idx:
                            j += 1

                        if j < len(endofchunk_token_ids_all):
                            endofchunk_token_idx = endofchunk_token_ids_all[j]
                            labels[i, answer_token_idx + 1 : endofchunk_token_idx + 1] = input_ids[i, answer_token_idx + 1 : endofchunk_token_idx + 1]

                            # Increment j for the next iteration
                            j += 1

                    for answer_token_idx, endofchunk_token_idx in zip(answer_token_ids_all, endofchunk_token_ids_all):
                        labels[i, answer_token_idx + 1 : endofchunk_token_idx + 1] = input_ids[i, answer_token_idx + 1 : endofchunk_token_idx + 1]

                labels[:, 0] = masking_number
                if args.model_name == "idefics" and fake_token_image_exists:
                    labels[labels == fake_token_image_token_id] = masking_number

                return labels

            labels = masking()

            if args.remove_answer_token:
                input_ids, labels, attention_mask = find_and_remove_tokens(input_ids, labels, attention_mask, answer_token_id, tokenizer)  # find and remove certain tokens from input_ids, labels, and attention_mask

            if args.remove_eos_token:
                input_ids, labels, attention_mask = find_and_remove_tokens(input_ids, labels, attention_mask, endofchunk_token_id, tokenizer)

        with accelerator.accumulate(model):
            if num_steps == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                master_print(f"model: {unwrapped_model.__class__.__name__}")
                master_print(f"model dtype: {unwrapped_model.dtype if hasattr(unwrapped_model, 'dtype') else 'None'}")

            loss_mimicit = forward_pass(
                args,
                model,
                tokenizer,
                images,
                input_ids,
                attention_mask,
                labels,
                device_id,
                autocast_type,
                batch_mimicit,
            )

            if accelerator.mixed_precision == "fp16":
                accelerator.backward(loss_mimicit.to(device_id))
            else:
                accelerator.backward(loss_mimicit)

            #### BACKWARD PASS ####
            mean_loss = loss_mimicit.detach().mean()
            cur_batch_max_tokens = input_ids.shape[1]

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
            if accelerator.sync_gradients and args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                mimicit_samples_per_second = args.gradient_accumulation_steps * args.batch_size * args.world_size / step_time_m.sum
                mimicit_samples_per_second_per_gpu = args.gradient_accumulation_steps * args.batch_size / step_time_m.sum
                step_time_m.reset()
                data_time_m.reset()

                group_name = batch_mimicit["task_group"][0]
                assert all(item == group_name for item in batch_mimicit["task_group"]), "Not all items in the list are the same"
                if args.report_to_wandb:
                    wandb.log(
                        {
                            "data_time": data_time_m.avg,
                            "step_time": step_time_m.avg,
                            "max_tokens": cur_batch_max_tokens,
                            "mimicit_samples_per_second": mimicit_samples_per_second,
                            "mimicit_samples_per_second_per_gpu": mimicit_samples_per_second_per_gpu,
                            "lr": optimizer.param_groups[0]["lr"],
                            "loss_mimicit": mean_loss,
                            "global_step": global_step // args.gradient_accumulation_steps,
                            group_name: mean_loss,
                        },
                        commit=True,
                    )

            delete_tensors_from_dict(batch_mimicit)
            delete_tensors_from_dict(
                {
                    "other": [
                        images,
                        input_ids,
                        attention_mask,
                        labels,
                    ]
                }
            )

        if args.rank == 0 and global_step != 0 and (args.save_steps_interval != -1) and (global_step % args.save_steps_interval == 0):
            save_checkpoint(epoch=None, global_step=global_step, model=model, args=args, accelerator=accelerator)

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss MIMIC-IT: {mean_loss.item():.3f}")
            # reset to avoid CPU oom
            loss_mimicit = None
            batch_mimicit = None
            gc.collect()
            torch.cuda.empty_cache()

    del unwrapped_model


def main():
    args = parse_args()
    verify_yaml(args)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size

    device_id = accelerator.device

    if args.pretrained_model_name_or_path is not None:
        master_print(f"Loading pretrained model from {args.pretrained_model_name_or_path}")
        device_map = {"": device_id} if accelerator.distributed_type == "MULTI_GPU" or accelerator.distributed_type == "DEEPSPEED" else "auto"
        kwargs = {"local_files_only": args.offline, "device_map": device_map}

        if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
            kwargs.pop("device_map")

        if args.customized_config is not None:
            kwargs["config"] = args.customized_config

        if args.model_name.lower() == "otter":
            model = OtterForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                **kwargs,
            )
            args.tokenizer = model.text_tokenizer
            tokenizer = model.text_tokenizer
            image_processor = CLIPImageProcessor()

        elif args.model_name.lower() == "flamingo":
            model = FlamingoForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                **kwargs,
            )
            # add special tokens for instruction tuning
            model.text_tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>"]})
            model.config.update(
                {
                    "special_tokens": model.text_tokenizer.all_special_tokens,
                    "architectures": "OtterForConditionalGeneration",
                }
            )
            tokenizer = args.tokenizer = model.text_tokenizer
            image_processor = CLIPImageProcessor()
            # if not accelerator.distributed_type == "DEEPSPEED" or not accelerator.state.deepspeed_plugin.zero_stage == 3:
            # new_embedding_size = (len(model.text_tokenizer) // 64 + 1) * 64
            # master_print(f"Resizing Flamingo embedding from {len(model.text_tokenizer)} to {new_embedding_size}")
            # model.resize_token_embeddings(new_embedding_size, pad_to_multiple_of=64)

        elif args.model_name.lower() == "idefics":
            model = IdeficsForVisionText2Text.from_pretrained(
                args.pretrained_model_name_or_path,
                **kwargs,
            )
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()

            processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path, legacy=False)
            if "<answer>" not in processor.tokenizer.special_tokens_map["additional_special_tokens"]:
                past_special_tokens = processor.tokenizer.special_tokens_map["additional_special_tokens"]
                processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>"] + past_special_tokens})

            image_processor = args.image_processor = processor.image_processor
            tokenizer = args.tokenizer = processor.tokenizer
            # make embedding size divisible by 64 for hardware compatiblity https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            # resize_token_embedding is not for parameter sharing in deepspeed !!!!
        elif args.model_name.lower() == "llama2":
            model = LlamaForCausalLM.from_pretrained(
                args.pretrained_model_name_or_path,
                **kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
            past_special_tokens = tokenizer.special_tokens_map["additional_special_tokens"] if "additional_special_tokens" in tokenizer.special_tokens_map else [value for key, value in tokenizer.special_tokens_map.items()]
            if "<answer>" not in past_special_tokens:
                tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>", "<image>", "<|endofchunk|>"]})

            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "<PAD>"})

            args.tokenizer = tokenizer
            image_processor = None

        elif args.model_name.lower() == "fuyu":
            tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
            image_processor = FuyuImageProcessor()
            processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
            # Peiyuan: should rewrite this portion. Right now too ugly.
            image_processor = processor
            model = FuyuForCausalLM.from_pretrained(args.pretrained_model_name_or_path, **kwargs)
            args.processor = processor

            if args.enable_lora:
                if args.peft_model_name_or_path:
                    master_print(f"Loading finetuned LoRA model from {args.peft_model_name_or_path}")
                    model = PeftModel.from_pretrained(model, args.peft_model_name_or_path)
                    model = model.merge_and_unload()

                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h", "lm_head"],
                )
                master_print(f"Init LoRA model with config {lora_config}")
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()

        elif args.model_name.lower() == "debug_model":
            model = torch.nn.Linear(100, 100)
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

            tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>", "<image>", "<|endofchunk|>"]})
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "<PAD>"})

            image_processor = None

    if args.resize_embedding and hasattr(model, "lang_encoder") and "LlamaForCausalLM" in model.lang_encoder.__class__.__name__:
        model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))
        master_print(f"Resizing Llama embedding to {len(model.text_tokenizer)}")

    if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
        params_to_gather = [p for name, p in model.named_parameters() if p.requires_grad]
        with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
            master_print(device_id, f"Zero3 Optimization: Trainable Params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e9:.3f} B")

    if args.trained_ckpt is not None:
        train_ckpt = torch.load(args.trained_ckpt, map_location="cpu")
        if train_ckpt.get("model_state_dict", None) is not None:
            train_ckpt = train_ckpt["model_state_dict"]
        _ = model.load_state_dict(train_ckpt, strict=False)
        print(_[1])

    accelerator.wait_for_everyone()

    args.distributed_type = accelerator.distributed_type

    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")

    if args.rank == 0 and args.report_to_wandb:
        master_print(f"Logging to wandb as {args.wandb_entity}/{args.wandb_project}/{args.run_name}")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
        )

    mimicit_loaders = get_data(args, image_processor, tokenizer, "mimicit")
    total_training_steps = sum(len(dataloader) for dataloader in mimicit_loaders) * args.num_epochs
    resume_from_epoch = 0
    args.external_save_dir = os.path.join(args.external_save_dir, args.run_name) if args.external_save_dir else args.run_name

    optimizer = torch.optim.AdamW(get_grouped_params(model, wd=args.weight_decay), lr=args.learning_rate)

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    args.warmup_steps = total_training_steps * args.warmup_steps_ratio if args.warmup_steps_ratio is not None else args.warmup_steps
    args.total_training_steps = total_training_steps

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_training_steps,
        )
    elif args.lr_scheduler == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    if args.rank == 0 and args.report_to_wandb:
        wandb.config.update(vars(args))

    if accelerator.distributed_type == "DEEPSPEED" or accelerator.distributed_type == "MULTI_GPU":
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model, optimizer, lr_scheduler, mimicit_loaders = accelerator.prepare(model, optimizer, lr_scheduler, mimicit_loaders)

    model.train()
    # Main Training Loop
    for epoch in range(resume_from_epoch, args.num_epochs):
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
            # save_checkpoint(epoch, model, args, accelerator)
            save_final_weights(
                model,
                args,
                accelerator,
                processor=processor if "idefics" in args.model_name.lower() or "fuyu" in args.model_name.lower() else None,
                tokenizer=tokenizer if "llama2" in args.model_name.lower() else None,
            )
            master_print(f"Saved checkpoint at epoch {epoch+1}.")
        accelerator.wait_for_everyone()

    # Save the final weights
    save_final_weights(
        model,
        args,
        accelerator,
        processor=processor if "idefics" in args.model_name.lower() or "fuyu" in args.model_name.lower() else None,
        tokenizer=tokenizer if "llama2" in args.model_name.lower() else None,
    )
    # accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
