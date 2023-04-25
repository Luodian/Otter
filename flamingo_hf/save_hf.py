import argparse
import torch
from flamingo_hf.configuration_flamingo import FlamingoConfig
from flamingo_hf.modeling_flamingo import FlamingoModel
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, default="flamingo_hf/checkpoint.pt"
    )
    parser.add_argument(
        "--save-dir", type=str, required=True
    )
    parser.add_argument(
        "--add-answer-token", action="store_true"
    )
    args = parser.parse_args()
    config = FlamingoConfig.from_json_file("flamingo_hf/config.json")
    model = FlamingoModel(config)
    if args.add_answer_token:
        model.text_tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]}
                )
        model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)
    if not os.path.exists(os.path.dirname(args.save_dir)):
        os.makedirs(os.path.dirname(args.save_dir))
    model.save_pretrained(args.save_dir)
