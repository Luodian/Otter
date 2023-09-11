""" Main training script """

import glob
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import deepspeed
import numpy as np
import torch
import torch.nn
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import Adafactor, AutoProcessor, CLIPImageProcessor, HfArgumentParser, pipeline
from trl.core import LengthSampler
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead

from otter_ai import FlamingoForConditionalGeneration
from pipeline.train.data import get_data
from pipeline.utils.otter_ppo_trainer import OTTERPPOTrainer, PPOConfig, create_reference_model
from src import OtterForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    pretrained_model_name_or_path: Optional[str] = field(default="", metadata={"help": "the model name"})
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=256, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the number of gradient accumulation steps"})
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    rank: Optional[int] = field(default=0, metadata={"help": "the rank"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    worker: Optional[int] = field(default=24, metadata={"help": "the worker name"})
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    offline: Optional[bool] = field(default=False, metadata={"help": "whether to load from local files"})
    tracker_project_name: Optional[str] = field(default="otter_trl", metadata={"help": "the tracker project name"})


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(tokenizer, dataset_name="lvwerra/stack-exchange-paired", num_proc=24):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
    ds = ds.select(range(100))
    original_columns = ds.column_names
    num_proc = num_proc

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True, max_length=2048)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    # ds = ds.filter(lambda x: len(x["input_ids"]) < 256, batched=False)

    ds.set_format(type="torch")
    return ds


def main():
    parser = HfArgumentParser(ScriptArguments)
    args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    reward_model_name = args.reward_model_name
    config = PPOConfig(
        steps=args.steps,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        log_with=args.log_with,
        tracker_project_name=args.tracker_project_name,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=args.early_stopping,
        target_kl=args.target_kl,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        init_kl_coef=args.init_kl_coef,
        adap_kl_ctrl=args.adap_kl_ctrl,
        remove_unused_columns=False,
    )

    # train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
    # train_dataset = train_dataset.select(range(100000))
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size

    device_id = accelerator.local_process_index

    # Model Preparation
    if args.pretrained_model_name_or_path is not None:
        accelerator.print(f"Loading pretrained model from {args.pretrained_model_name_or_path}")
        device_map = {"": device_id} if accelerator.distributed_type == "MULTI_GPU" or accelerator.distributed_type == "DEEPSPEED" else "auto"
        if "otter" in args.model_name.lower():
            model = OtterForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                device_map=device_map,
                local_files_only=args.offline,
            )
            args.tokenizer = model.text_tokenizer
            tokenizer = model.text_tokenizer
        elif "flamingo" in args.model_name.lower():
            model = FlamingoForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                device_map=device_map,
                local_files_only=args.offline,
            )
            # add special tokens for instruction tuning
            model.text_tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>"]})
            args.tokenizer = model.text_tokenizer
            tokenizer = model.text_tokenizer
        elif "idefics" in args.model_name.lower():
            from transformers import IdeficsForVisionText2Text

            # you need to install the idefics version transformers package first
            kwargs = {"local_files_only": args.offline, "device_map": device_map}
            if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
                kwargs.pop("device_map")

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
                del params_to_gather

            print(
                device_id,
                f"IDEFICS Trainable Params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e9:.3f} B",
            )
            # import pdb;pdb.set_trace()
            processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path, legacy=False)
            past_special_tokens = processor.tokenizer.special_tokens_map["additional_special_tokens"]
            processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>", "<|endofchunk|>"] + past_special_tokens})
            image_processor = processor.image_processor
            tokenizer = processor.tokenizer
            # For idefics model, do not resize token
            # model.resize_token_embeddings(len(tokenizer))

    accelerator.wait_for_everyone()

    args.distributed_type = accelerator.distributed_type

    if hasattr(model, "lang_encoder") and "LlamaForCausalLM" in model.lang_encoder.__class__.__name__:
        model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))

    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 512,
    }

    optimizer = None
    if args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(tokenizer, num_proc=args.worker)

    model.lang_encoder_vhead = AutoModelForCausalLMWithValueHead(model.lang_encoder)

    # We then build the OTTERPPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = OTTERPPOTrainer(
        config,
        model=model,
        # ref_model=ref_lang_encoder,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
        "truncation": True,
    }

    # We then build the sentiment analysis pipeline using our reward model, passing the
    # model name and the sentiment analysis pipeline arguments. Let's also make sure to
    # set the device to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model_name,
        device_map={"": device_id},
        model_kwargs={"load_in_8bit": True},
        tokenizer=tokenizer,
        return_token_type_ids=False,
    )

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000,
    }
    output_min_length = 32
    output_max_length = args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if epoch >= config.total_ppo_epochs:
            break

        question_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute reward score (using the sentiment analysis pipeline)
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"] - args.reward_baseline) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if args.save_freq and epoch and epoch % args.save_freq == 0:
            ppo_trainer.save_pretrained(args.output_dir + f"step_{epoch}")


if __name__ == "__main__":
    main()
