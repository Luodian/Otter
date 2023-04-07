from open_flamingo import create_model_and_transforms
from peft.src.peft import LoraModel, LoraConfig
import torch

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14", clip_vision_encoder_pretrained="openai", lang_encoder_path="./llama-7b-hf", tokenizer_path="./llama-7b-hf", cross_attn_every_n_layers=4
)

checkpoint_path = "/home/v-boli7/azure_storage/models/openflamingo/checkpoint.pt"
model.load_state_dict(torch.load(checkpoint_path), strict=False)

config = LoraConfig(
    peft_type="LORA",
    task_type="SEQ_2_SEQ_LM",
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.01,
)

lora_model = LoraModel(config, model)


total_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")
