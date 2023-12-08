import numpy as np
import torch
import torchvision.transforms as T
from torchvision.io import read_video

from .base_model import BaseModel
from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .llava.conversation import conv_templates, SeparatorStyle
from .llava.model.builder import load_pretrained_model
from .llava.utils import disable_torch_init
from .llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

default_model_path = "liuhaotian/llava-v1.5-7b"


class LLaVA_Model(BaseModel):
    def __init__(
        self,
        model_path: str = default_model_path,
        model_base: str = None,
        model_name: str = "llava-v1.5",
        conv_mode: str = "llava_v1",
    ):
        super().__init__(model_name, model_path)
        init_model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, init_model_name)
        self.conv_mode = conv_mode

    def generate(self, text_prompt: str, raw_image_data: str):
        if self.model.config.mm_use_im_start_end:
            prompts_input = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text_prompt
        else:
            prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt

        input_data = self.image_processor.preprocess(raw_image_data, return_tensors="pt")["pixel_values"][0]

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompts_input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=input_data.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=512,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        return outputs

    def eval_forward(self, text_prompt: str, raw_image_data: str):
        pass
