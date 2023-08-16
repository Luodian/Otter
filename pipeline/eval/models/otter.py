from typing import List

from PIL import Image
import torch
import transformers

from pipeline.eval.eval_model import BaseEvalModel
from contextlib import suppress
from pipeline.eval.models.utils import unwrap_model
from otter.modeling_otter import OtterForConditionalGeneration
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EvalModel(BaseEvalModel):
    """OpenFlamingo model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(self, model_args):
        def get_precision(load_bit: str):
            if load_bit == "fp16":
                return torch.float16
            elif load_bit == "bf16":
                return torch.bfloat16
            elif load_bit == "fp32":
                return torch.float32

        self.device = model_args["device"]
        kwargs = {"torch_dtype": get_precision(model_args["precision"])}
        self.model = OtterForConditionalGeneration.from_pretrained(
            model_args["model_path"],
            **kwargs,
        ).cuda()
        # self.model.to(self.device)
        self.image_processor = transformers.CLIPImageProcessor()
        self.tokenizer = self.model.text_tokenizer

        if "checkpoint_path" in model_args:
            checkpoint = torch.load(model_args["checkpoint_path"], map_location=self.device)
            if "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            msg = self.model.load_state_dict(checkpoint, strict=False)
            print(msg)
        # self.model.to(self.device)
        self.model.eval()
        self.tokenizer.padding_side = "left"

        # autocast
        self.autocast = get_autocast(model_args["precision"])
        self.cast_dtype = get_cast_dtype(model_args["precision"])

    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = torch.from_numpy(self.image_processor(image)["pixel_values"][0])

                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        return batch_images

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2048,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            with self.autocast():
                outputs = unwrap_model(self.model).generate(
                    self._prepare_images(batch_images).to(self.device, dtype=self.cast_dtype, non_blocking=True),
                    input_ids.to(self.device, non_blocking=True),
                    attention_mask=attention_mask.to(self.device, non_blocking=True),
                    min_new_tokens=min_generation_length,
                    max_new_tokens=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_logits(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
        clear_conditioned_layers: bool = False,
    ):
        with torch.inference_mode():
            with self.autocast():
                outputs = self.model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    clear_conditioned_layers=clear_conditioned_layers,
                    past_key_values=past_key_values,
                    use_cache=(past_key_values is not None),
                )
        return outputs

    def encode_vision_x(self, image_tensor: torch.Tensor):
        unwrap_model(self.model)._encode_vision_x(image_tensor.to(self.device))

    def uncache_media(self):
        unwrap_model(self.model).uncache_media()

    def cache_media(self, input_ids, vision_x):
        unwrap_model(self.model).cache_media(input_ids=input_ids, vision_x=vision_x)

    def get_vqa_prompt(self, question, answer=None) -> str:
        return f"<image>User: {question} Please answer in short words. GPT:<answer>{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_caption_prompt(self, caption=None) -> str:
        return (
            f"<image>User: What does the image describe? GPT:<answer>{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"
        )


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
