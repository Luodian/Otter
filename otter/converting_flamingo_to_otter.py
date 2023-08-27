# This script is used to convert the huggingface format Open-Flamingo model to the Otter model.
# You can use it in parent folder by running: python -m otter.converting_flamingo_to_otter --checkpoint_path <path_to_flamingo_checkpoint> --save_path <path_to_save_otter_checkpoint>
import argparse
import torch
from otter.modeling_otter import OtterForConditionalGeneration
from flamingo.modeling_flamingo import FlamingoForConditionalGeneration

# Define argument parser
parser = argparse.ArgumentParser(description="Load a model with specified precision and save it to a specified path.")

# Add arguments
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the pre-trained Open-Flamingo model checkpoint.")
parser.add_argument("--save_path", type=str, default=None, help="Path to the converted Otter model checkpoint.")

# Parse the input arguments
args = parser.parse_args()

# Load the model
model = FlamingoForConditionalGeneration.from_pretrained(args.checkpoint_path, device_map="auto")
model.text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]})
if model.lang_encoder.__class__.__name__ == "LlamaForCausalLM":
    model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))

# Save the model
checkpoint_path = args.save_path
OtterForConditionalGeneration.save_pretrained(model, checkpoint_path)
