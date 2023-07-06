import argparse
import torch
from otter.modeling_otter import OtterForConditionalGeneration

# Define argument parser
parser = argparse.ArgumentParser(description="Load a model with specified precision and save it to a specified path.")

# Add arguments
parser.add_argument(
    "--load_bit",
    type=str,
    choices=["fp16", "bf16"],
    default="fp16",
    help="Precision of the loaded model. Either 'fp16' or 'bf16'. Default is 'fp16'.",
)
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the pre-trained model checkpoint.")
parser.add_argument("--save_path", type=str, default=None, help="Path to the converted model checkpoint.")

# Parse the input arguments
args = parser.parse_args()

# Set precision based on load_bit argument
if args.load_bit == "fp16":
    precision = {"torch_dtype": torch.float16}
elif args.load_bit == "bf16":
    precision = {"torch_dtype": torch.bfloat16}

# Load the model
model = OtterForConditionalGeneration.from_pretrained(args.checkpoint_path, device_map="auto", **precision)

# Save the model
if args.save_path is None:
    checkpoint_path = args.checkpoint_path + f"-{args.load_bit}"
else:
    checkpoint_path = args.save_path
OtterForConditionalGeneration.save_pretrained(model, checkpoint_path)
