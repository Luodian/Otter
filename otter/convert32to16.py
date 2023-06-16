from otter.modeling_otter import OtterForConditionalGeneration
import torch

load_bit = "fp16"

if load_bit == "fp16":
    precision = {"torch_dtype": torch.float16}
elif load_bit == "bf16":
    precision = {"torch_dtype": torch.bfloat16}

# checkpoint_path = "checkpoint/otter9B_LA_incontext2"
checkpoint_path = "luodian/otter-9b-hf"
model = OtterForConditionalGeneration.from_pretrained(checkpoint_path, device_map="auto", **precision)

# save model
checkpoint_path = checkpoint_path + f"_{load_bit}"
OtterForConditionalGeneration.save_pretrained(model, checkpoint_path)
