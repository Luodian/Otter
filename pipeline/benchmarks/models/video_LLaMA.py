import yaml
import numpy as np
import argparse

from .base_model import BaseModel
from .video_llama.common.registry import registry
from .video_llama.common.config import Config
from .video_llama.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2

class VideoLLaMA(BaseModel):
    def __init__(self, model_name: str, model_path: str, model_config_path, with_AL : bool = True):
        super().__init__(model_name, model_path)
        self.with_AL = with_AL
        args = {'cfg_path' : model_config_path, 'options' : None}
        config = Config(args)
        
        model_config = config.model_cfg
        
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda')
        model.eval()
        
        vis_processor_cfg = config.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device = 'cuda')
        print("Initialization Complete")
        
    def generate(self, input_data: dict):
        questions = input_data["question"]
        video_dir = input_data.get("video_root", "")
        video_list = input_data["video_path"]
        
        chat_state = conv_llava_llama_2.copy()
        img_list = []
        attn_list = []
        
        if self.with_AL:
            video, audio = self.chat.upload_video(video_list, chat_state, img_list, attn_list)
        else:
            video, audio = self.chat.upload_video_without_audio(video_list, chat_state, img_list, attn_list)
        prompt = input_data['prompt']
        self.chat.ask(prompt, chat_state)
        
        if input_data['task'] != 'MC_PPL':
            llm_message = self.chat.answer(
                conv = chat_state,
                img_list=img_list,
            )[0]
        else:
            samples = {}
            samples['video'] = video
            samples['audio'] = audio
            all_choice_losses = []
            for cur_option in input_data['option']:
                samples['text_input'] = [prompt + cur_option.split(".")[1] + "."]
            
                output = self.chat.forward(samples)
                cur_loss = output['loss']
                all_choice_losses.append(cur_loss.item())
            llm_message = input_data["option"][np.argmin(all_choice_losses)].split(".")[0]
            
            
        print(f"Question : {prompt}")
        print(f"Answer : {llm_message}")
        return llm_message
    
    def to(self, device: str):
        pass