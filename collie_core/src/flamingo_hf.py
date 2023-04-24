from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers import LlamaTokenizer

from open_flamingo.src.flamingo import Flamingo
from open_flamingo.src.factory import _infer_decoder_layers_attr_name
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
import open_clip

class FlamingoConfig(PretrainedConfig):
    model_type = "flamingo"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FlamingoPreTrainedModel(PreTrainedModel):
    config_class = FlamingoConfig

    def __init__(self, config: FlamingoConfig):
        super().__init__(config)
        self.vision_encoder, _, self.image_processor = open_clip.create_model_and_transforms(
            config.vision_encoder_path,
            pretrained=config.vision_encoder_pretrained,
        )
        # set the vision encoder to output the visual features
        self.vision_encoder.visual.output_tokens = True

        self.text_tokenizer = LlamaTokenizer.from_pretrained(
            config.tokenizer_path, local_files_only=bool(config.use_local_files)
        )
        # add Flamingo special tokens to the tokenizer
        self.text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]},
        )
        if self.text_tokenizer.pad_token is None:
            # Issue: GPT models don't have a pad token, which we use to
            # modify labels for the loss.
            self.text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        self.lang_encoder = AutoModelForCausalLM.from_pretrained(
            config.lang_encoder_path,
            local_files_only=bool(config.use_local_files),
            # device_map='auto'
        )
        extend_instance(self.lang_encoder, FlamingoLMMixin)

        # if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(self.lang_encoder)
        self.lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        self.lang_encoder.resize_token_embeddings(len(self.text_tokenizer))

        self.model = Flamingo(
            self.vision_encoder,
            self.lang_encoder,
            self.text_tokenizer.encode("<|endofchunk|>")[-1],
            self.text_tokenizer.encode("<image>")[-1],
            vis_dim=open_clip.get_model_config(config.vision_encoder_path)[
                "vision_cfg"
            ]["width"],
            cross_attn_every_n_layers=config.cross_attn_every_n_layers,
        )
        
        # Freeze all parameters
        self.model.requires_grad_(False)
        assert sum(p.numel() for p in self.model.parameters() if p.requires_grad) == 0
        
        self.model.perceiver.requires_grad_(True)
        self.model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        self.model.lang_encoder.get_input_embeddings().requires_grad_(True)

        print(
            f"Flamingo model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters"
        )


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


if __name__ == "__main__":
    from accelerate import load_checkpoint_and_dispatch
    from accelerate import Accelerator
    accelerator = Accelerator()
    config = FlamingoConfig.from_json_file("./open_flamingo/src/config.json")
    model = FlamingoPreTrainedModel(config)
    from huggingface_hub import hf_hub_download
    import torch
    from PIL import Image
    import requests
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
    model.model.load_state_dict(torch.load(checkpoint_path), strict=False)
    
    """
    Step 1: Load images
    """
    demo_image_one = Image.open(
        requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
        ).raw
    )

    demo_image_two = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
            stream=True
        ).raw
    )

    query_image = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
            stream=True
        ).raw
    )


    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
    batch_size x num_media x num_frames x channels x height x width. 
    In this case batch_size = 1, num_media = 3, num_frames = 1 
    (this will always be one expect for video which we don't support yet), 
    channels = 3, height = 224, width = 224.
    """
    import torch

    vision_x = [model.image_processor(demo_image_one).unsqueeze(0), model.image_processor(demo_image_two).unsqueeze(0), model.image_processor(query_image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text 
    portion associated with an image.
    """
    model.text_tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    lang_x = model.text_tokenizer(
        ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
        return_tensors="pt",
    )


    """
    Step 4: Generate text
    """
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20,
        num_beams=3,
    )
    
    print(model.text_tokenizer.decode(generated_text[0]))
    
    model.save_pretrained("/media/ntu/volume2/s121md302_06/code/mutoo/PET-VLM/openflamingo_hf")
    # model = accelerator.prepare(model)
    # print(model.module.config)
    # # model = load_checkpoint_and_dispatch(model, "/media/ntu/volume1/home/s121md302_06/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-9B/snapshots/b5cd34cb6c90775b262837b6a80a6a47123b4571/checkpoint.pt")
    # from huggingface_hub import hf_hub_download
    # import torch
    # checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
    # model.load_state_dict(torch.load(checkpoint_path), strict=False)
    # print(model.module.config)
