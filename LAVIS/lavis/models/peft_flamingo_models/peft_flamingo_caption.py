"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip
from peft.src.peft import LoraModel, LoraConfig

from open_flamingo.src.flamingo import Flamingo
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
from open_flamingo.src.factory import _infer_decoder_layers_attr_name
from lavis.common.registry import registry

from lavis.models.peft_flamingo_models.peft_flamingo import PEFT_FLAMINGO
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)

@registry.register_model("peft_flamingo_caption")
class PEFT_FLAMINGO_Caption(PEFT_FLAMINGO):
    """
    PEFT FLAMINGO captioning model.

    Supported model types:
        - base_coco: fine-tuned BLIP base model on COCO caption dataset (Karparthy split).
        - large_coco: fine-tuned BLIP large model on COCO caption dataset (Karparthy split).

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_caption", "base_coco")
        >>> model = load_model("blip_caption", "large_coco")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base_coco": "configs/models/blip_caption_base_coco.yaml",
        "large_coco": "configs/models/blip_caption_large_coco.yaml",
    }

    def __init__(self, image_encoder, lang_encoder, text_tokenizer, 
                 clip_vision_encoder_path, cross_attn_every_n_layers=1, 
                 pretrained_checkpoint_path=None, max_txt_len=40, prompt=None):
        super().__init__()

        self.image_encoder = image_encoder
        self.lang_encoder = lang_encoder
        self.tokenizer = text_tokenizer

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        self.model = Flamingo(
            image_encoder,
            lang_encoder,
            text_tokenizer.encode("<|endofchunk|>")[-1],
            text_tokenizer.encode("<image>")[-1],
            vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
                "width"
            ],
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            # **flamingo_kwargs,
        )

        self.max_txt_len = max_txt_len

        # Freeze all parameters
        self.model.requires_grad_(False)
        assert sum(p.numel() for p in self.model.parameters() if p.requires_grad) == 0

        # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
        self.model.perceiver.requires_grad_(True)
        self.model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        self.model.lang_encoder.get_input_embeddings().requires_grad_(True)

        print(f"Flamingo model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6} M trainable parameters")
        print(f"Loading checkpoint from {pretrained_checkpoint_path}...")
        msg = self.model.load_state_dict(torch.load(pretrained_checkpoint_path), strict=False)
        # print(msg)
        self.model.device = torch.device("cuda")

        config = LoraConfig(
            peft_type="LORA",
            task_type="SEQ_2_SEQ_LM",
            r=8,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.01,
        )

        self.model = LoraModel(config, self.model)

        # total_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        # print(f"Total number of trainable parameters in LoRA is {total_params / 1e6}M")
        

    def forward_encoder(self, samples):
        image_embeds = self.visual_encoder.forward_features(samples["image"])
        return image_embeds

    def forward_decoder(self, samples, image_embeds):
        # prepare inputs for forwarding decoder
        raw_text = samples["text_input"]
        text = self.tokenizer(
            raw_text,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        # prepare targets for forwarding decoder
        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        # forward decoder
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        decoder_output = self.text_decoder(
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )

        return decoder_output, decoder_targets

    def forward(self, samples):
        r"""
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size.
        Returns:
            output (BlipOutput): A BlipOutput object containing the following
                attributes:
                - loss (torch.Tensor): A scalar tensor containing the total loss. For BlipCaption, this is the same as the LM loss.
                - loss_lm (torch.Tensor): A scalar tensor containing the LM loss.
                - intermediate_outputs (BlipIntermediateOutput): A BlipIntermediateOutput object containing intermediate outputs.
                  see :class:`lavis.models.blip_models.blip_outputs.BlipOutput` for more details.

        Example:
        ```python
        >>> from PIL import Image
        >>> from lavis.models import load_model_and_preprocess
        >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption")
        >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
        >>> text_input = ["a large statue of a person spraying water from a fountain"]
        >>> samples = {"image": image, "text_input": text_input}
        >>> output = model(samples)
        >>> output.keys()
        odict_keys(['intermediate_output', 'loss', 'loss_lm'])
        >>> output.intermediate_output.image_embeds.shape
        torch.Size([1, 577, 768])
        >>> output.intermediate_output.decoder_labels.shape
        torch.Size([1, 13])
        ```"""

        images = samples["image"].unsqueeze(1).unsqueeze(2)
        
        # image_embeds = self.image_encoder.forward_features(images)
        
        raw_text = samples["text_input"]
        text = self.tokenizer(raw_text, 
                              padding="longest", 
                              truncation=True, 
                              max_length=self.max_txt_len, 
                              return_tensors="pt").to(self.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        # prepare targets for forwarding decoder
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, : self.prompt_length] = -100

        loss = self.model(
            vision_x=images,
            lang_x=text["input_ids"],
            attention_mask=text["attention_mask"],
            labels=decoder_targets,
        )[0]

        output = {'loss': loss}

        # image_embeds = self.forward_encoder(samples)
        # decoder_output, decoder_targets = self.forward_decoder(samples, image_embeds)

        # return decoder_out
        return output

    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        num_captions=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.

        Example:
        ```python
        >>> from PIL import Image
        >>> from lavis.models import load_model_and_preprocess
        >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption")
        >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
        >>> samples = {"image": image}
        >>> captions = model.generate(samples)
        >>> captions
        ['a large statue of a person spraying water from a fountain']
        >>> captions = model.generate(samples, use_nucleus_sampling=True, num_captions=3)
        >>> captions # example output, results may vary due to randomness
        ['singapore showing the view of some building',
        'the singapore harbor in twilight, as the weather is going down',
        'the famous singapore fountain at sunset']
        """
        # prepare inputs for decoder generation.

        images = samples["image"].unsqueeze(1).unsqueeze(2)
        if num_beams > 1:
            images = images.repeat_interleave(num_beams, dim=0)

        self.model._encode_vision_x(images)
        # output = self.model.lang_encoder.generate()

        # get decoded text
        # decoder_out = self.text_decoder.generate_from_encoder(
        #     tokenized_prompt=prompt,
        #     visual_embeds=image_embeds,
        #     sep_token_id=self.tokenizer.sep_token_id,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     use_nucleus_sampling=use_nucleus_sampling,
        #     num_beams=num_beams,
        #     max_length=max_length,
        #     min_length=min_length,
        #     top_p=top_p,
        #     repetition_penalty=repetition_penalty,
        # )

        # outputs = self.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
        # captions = [output[len(self.prompt) :] for output in outputs]
        captions = None

        return captions

    @classmethod
    def from_config(cls, cfg):
        # vision encoder
        # image_encoder = VisionTransformerEncoder.from_config(cfg)
        clip_vision_encoder_path = cfg.get("clip_vision_encoder_path", None)
        clip_vision_encoder_pretrained = cfg.get("clip_vision_encoder_pretrained", True)
        tokenizer_path = cfg.get("tokenizer_path", None)
        use_local_files = cfg.get("use_local_files", False)
        lang_encoder_path = cfg.get("lang_encoder_path", None)
        cross_attn_every_n_layers = cfg.get("cross_attn_every_n_layers", 1)
        decoder_layers_attr_name = cfg.get("decoder_layers_attr_name", None)
        pretrained_checkpoint_path = cfg.get("pretrained_checkpoint_path", None)

        prompt = cfg.get("prompt", None)
        max_txt_len = cfg.get("max_txt_len", 40)
        
        image_encoder, _, image_processor = open_clip.create_model_and_transforms(clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained)
        # set the vision encoder to output the visual features
        image_encoder.visual.output_tokens = True
        
        text_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=use_local_files
        )
        # add Flamingo special tokens to the tokenizer
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
        if text_tokenizer.pad_token is None:
            # Issue: GPT models don't have a pad token, which we use to
            # modify labels for the loss.
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path, local_files_only=use_local_files
        )
        extend_instance(lang_encoder, FlamingoLMMixin)

        if decoder_layers_attr_name is None:
            decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        lang_encoder.resize_token_embeddings(len(text_tokenizer))

        model = cls(image_encoder, lang_encoder, text_tokenizer, clip_vision_encoder_path, cross_attn_every_n_layers, pretrained_checkpoint_path, prompt=prompt, max_txt_len=max_txt_len)

        return model