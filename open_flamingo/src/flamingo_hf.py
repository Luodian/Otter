from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

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

        self.text_tokenizer = AutoTokenizer.from_pretrained(
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

    def forward(self, input):
        return self.model(input)


# if __name__ == "__main__":
#     from accelerate import load_checkpoint_and_dispatch
#     from accelerate import Accelerator
#     accelerator = Accelerator()
#     config = FlamingoConfig.from_json_file("./open_flamingo/src/config.json")
#     model = FlamingoPreTrainedModel(config)
#     model = accelerator.prepare(model)
#     print(model.config)
    # mymodel = load_checkpoint_and_dispatch(mymodel, "./checkpoint.bt")
