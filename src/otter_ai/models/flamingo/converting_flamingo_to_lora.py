import argparse
import torch
import sys

from .modeling_flamingo import FlamingoForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType

MODEL_CLASSES = {
    "LlamaForCausalLM": "llama",
    "OPTForCausalLM": "opt",
    "GPTJForCausalLM": "gptj",
    "GPTNeoXForCausalLM": "gpt_neox",
    "MPTForCausalLM": "mpt",
}

# Define argument parser
parser = argparse.ArgumentParser(description="Load a model with specified precision and save it to a specified path.")

# Add arguments
parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Path to the pre-trained model checkpoint.",
    default="",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="",
    help="Path to the converted model checkpoint.",
)

# Parse the input arguments
args = parser.parse_args()

load_bit = "bf16"
if load_bit == "fp16":
    precision = {"torch_dtype": torch.float16}
elif load_bit == "bf16":
    precision = {"torch_dtype": torch.bfloat16}

# Load the model
model = FlamingoForConditionalGeneration.from_pretrained(args.checkpoint_path, device_map="auto", **precision)

# adding lora
standard_modules = ["q_proj", "v_proj"]
lang_encoder_short_name = MODEL_CLASSES[model.config.text_config.architectures[0]]
model_to_lora_modules = {
    "llama": standard_modules,
    "opt": standard_modules,
    "gptj": standard_modules,
    "gpt_neox": ["query_key_value"],
    "mpt": ["Wqkv"],
}
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules=model_to_lora_modules[lang_encoder_short_name],
)
model.config.update({"lora_config": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}})
model.lang_encoder = get_peft_model(model.lang_encoder, lora_config)
model.lang_encoder.print_trainable_parameters()

# Save the model
checkpoint_path = args.save_path
FlamingoForConditionalGeneration.save_pretrained(model, checkpoint_path)
