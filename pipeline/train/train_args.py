import argparse
import os

from pipeline.train.distributed import world_info_from_env


def parse_tuple(string):
    try:
        x, y = map(int, string.split(","))
        return (x, y)
    except:
        raise argparse.ArgumentTypeError("Invalid tuple format. Expected 'x,y'")


def parse_args():
    """
    Parse the command line arguments and perform the initial setup.
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Main training script for the model")
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
        choices=["otter", "flamingo", "idefics", "llama2", "debug_model", "fuyu"],
        help="otters or flamingo",
    )
    parser.add_argument(
        "--instruction_format",
        type=str,
        default="simple",
        choices=["simple", "llama2", "idefics", "fuyu"],
        help="simple is for mpt/llama1, rest are in different instruction templates.",
    )
    parser.add_argument(
        "--training_data_yaml",
        type=str,
        default="",
        help="Path to the training data yaml file.",
    )

    # optimizer args
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--save_ckpt_each_epoch", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100, help="log loss every n steps")
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps_interval", type=int, default=-1)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="path to huggingface model or model identifier from local path or huggingface.co",
        default=None,
    )
    parser.add_argument(
        "--peft_model_name_or_path",
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
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
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
    ),
    parser.add_argument(
        "--keep_symbols",
        action="store_true",
        default=False,
        help="keep symbols in the generated text",
    )
    parser.add_argument(
        "--remove_answer_token",
        action="store_true",
        default=False,
        help="we have an <answer> token as indicator for separating question and answer, use this flag to remove it before training.",
    )
    parser.add_argument(
        "--remove_eos_token",
        action="store_true",
        default=False,
        help="we have an eos token as indicator for separating question and answer, use this flag to remove it before training.",
    )
    parser.add_argument(
        "--populate_rel_ins",
        action="store_true",
        default=False,
        help="populate rel_ins into train_config.",
    )
    parser.add_argument(
        "--resize_embedding",
        action="store_true",
        default=False,
        help="resize embedding layer to match the vocabulary size.",
    )
    parser.add_argument("--image_resolution", type=parse_tuple, default=(224, 224), help="image resolution for the model in format: x,y")
    parser.add_argument(
        "--with_task_description",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--enable_lora",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dynamic_resolution",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    # Check for argument consistency and set environment variables if needed
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    return args
