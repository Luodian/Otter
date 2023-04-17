import torch
from collections import OrderedDict,abc
from tqdm import tqdm
from torch import nn
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
import os, random, json
import numpy as np
import logging
from typing import Optional, Dict, Union
from .presets import *
from .configurations import TrainingConfig, DistillationConfig
import random
from .compatibility import mask_dtype, is_apex_available

has_apex = is_apex_available()
if has_apex:
    from apex import amp


logger = logging.getLogger("Distillation")
#logger.setLevel(logging.INFO)

#handler_stream = logging.StreamHandler()
#handler_stream.setLevel(logging.INFO)
#formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -  %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
#handler_stream.setFormatter(formatter)
#logger.addHandler(handler_stream)

class CustomMatch:
    def __init__(self, module_T, module_S, weight, loss,
                 proj_func =None, proj_group = None):
        self.module_T = module_T
        self.module_S = module_S
        self.loss     = loss,
        self.weight   = weight,
        self.proj_func     = proj_func
        if proj_group is None:
            self.proj_group = dict()
        else:
            self.proj_group = proj_group
    def to_dict(self):
        return {'module_T':self.module_T,
                'module_S':self.module_S,
                'weight':self.weight,
                'loss':self.loss,
                'proj_func':self.proj_func,
                'proj_group':self.proj_group}
    @classmethod
    def from_dict(cls,dict_object):
        return cls(**dict_object)


class DistillationContext:
    def __init__(self):
        self.model_S = None
        self.model_T = None
    def __enter__(self):
        if isinstance(self.model_T,(list,tuple)):
            self.model_T_is_training = [model_t.training for model_t in self.model_T]
            for model_t in self.model_T:
                model_t.eval()
        elif isinstance(self.model_T,dict):
            self.model_T_is_training = {name:model.training for name,model in self.model_T.items()}
            for name in self.model_T:
                self.model_T[name].eval()
        else:
            self.model_T_is_training = self.model_T.training
            self.model_T.eval()

        if isinstance(self.model_S,(list,tuple)):
            self.model_S_is_training = [model_s.training for model_s in self.model_S]
            for model_s in self.model_S:
                model_s.eval()
        elif isinstance(self.model_S,dict):
            self.model_S_is_training = {name:model.training for name,model in self.model_S.items()}
            for name in self.model_S:
                self.model_S[name].eval()
        else:
            self.model_S_is_training = self.model_S.training
            self.model_S.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        #Restore model status
        if isinstance(self.model_T,(list,tuple)):
            for i in range(len(self.model_T_is_training)):
                self.model_T[i].train(self.model_T_is_training[i])
        elif isinstance(self.model_T,dict):
            for name,is_training  in self.model_T_is_training.items():
                self.model_T[name].train(is_training)
        else:
            self.model_T.train(self.model_T_is_training)

        if isinstance(self.model_S,(list,tuple)):
            for i in range(len(self.model_S_is_training)):
                self.model_S[i].train(self.model_S_is_training[i])
        elif isinstance(self.model_S,dict):
            for name,is_training  in self.model_S_is_training.items():
                self.model_S[name].train(is_training)
        else:
            self.model_S.train(self.model_S_is_training)


class AbstractDistiller(DistillationContext):
    def __init__(self, train_config: TrainingConfig,
                       distill_config: DistillationConfig,
                       model_T, model_S, adaptor_T, adaptor_S):
        super(AbstractDistiller, self).__init__()
        self.t_config = train_config
        self.d_config = distill_config

        self.model_T = model_T
        self.model_S = model_S
        self.adaptor_S = adaptor_S
        self.adaptor_T = adaptor_T

        self.kd_loss = KD_LOSS_MAP[self.d_config.kd_loss_type]

        self.local_rank = self.t_config.local_rank
        self.rank = 0
        if self.local_rank != -1:
            self.rank = torch.distributed.get_rank()
        if self.t_config.log_dir is not None and self.rank == 0:
            self.tb_writer = SummaryWriter(log_dir = self.t_config.log_dir)
        else:
            self.tb_writer = no_op
        
        self.print_freq = 20

        self.logits_cache = []


def select_logits_with_mask(logits_list, masks_list):
    output_logits = []
    if len(masks_list)==len(logits_list):
        for logits,mask in zip(logits_list,masks_list):
            if len(logits.shape)==3:
                mask = mask.unsqueeze(-1).expand_as(logits).to(mask_dtype)
                logits_select = torch.masked_select(logits,mask).view(-1,logits.size(-1))
            else:
                logits_select = logits #Logits_mask has no effect on logits of shape (batch_size, logits_to_be_softmaxed)
            output_logits.append(logits_select)
    elif len(masks_list)==1:
        mask = masks_list[0]
        for logits in logits_list:
            if len(logits.shape)==3:
                mask = mask.unsqueeze(-1).expand_as(logits).to(mask_dtype)
                logits_select = torch.masked_select(logits,mask).view(-1,logits.size(-1))
            else:
                logits_select = logits #Logits_mask has no effect on logits of shape (batch_size, logits_to_be_softmaxed)
            output_logits.append(logits_select)
    else:
        raise AssertionError("lengths of logits list and masks list mismatch")
    return output_logits


class BasicAdaptor:
    def __init__(self):
        self.batch = None
        self.model_outputs = None
    def __call__(self,batch,model_outputs):
        self.batch = batch
        self.model_outputs = model_outputs
    def __getattr__(self, item):
        raise NotImplementedError


def post_adaptor(dict_object):
    if 'logits' in dict_object:
        logits = dict_object['logits']
        if not isinstance(logits,(list,tuple)):
            dict_object['logits'] = [ logits ]
    if 'logits_mask' in dict_object:
        logits_mask = dict_object['logits_mask']
        if not isinstance(logits_mask,(list,tuple)):
            dict_object['logits_mask'] = [ logits_mask ]
    if 'losses' in dict_object:
        losses = dict_object['losses']
        if not isinstance(losses,(list,tuple)):
            dict_object['losses'] = [ losses ]
    if 'labels' in dict_object:
        labels = dict_object['labels']
        if not isinstance(labels,(list,tuple)):
            dict_object['labels'] = [ labels ]
    return dict_object


def probability_shift_(tensor, labels):  # In-place operation. shape (batch_size, num_classes), (batch_size,)
    if len(tensor.shape)==2:
        max_position = tensor.argmax(dim=-1) # shape (batch_size,)
        index = torch.arange(tensor.size(0)).to(tensor.device)
        max_clone = tensor[index,max_position].clone()
        truth_clone = tensor[index,labels].clone()

        tensor[index,max_position] = truth_clone
        tensor[index,labels] = max_clone
        return tensor

    elif len(tensor.shape)==3:   # shape (batch_size, length, num_classes)
        original_shape = tensor.size()

        tensor = tensor.view(-1,tensor.size(-1))   # (batch_size * length, num_classes)

        max_position = tensor.argmax(dim=-1) # shape (batch_size * length, )
        labels = labels.view(-1) # (batch_size * length, )
        nonneg_labels = torch.where(labels<0, max_position, labels)

        index = torch.arange(tensor.size(0)).to(tensor.device)   # (batch_size * length)

        max_clone = tensor[index,max_position].clone()
        truth_clone = tensor[index,nonneg_labels].clone()

        tensor[index,max_position] = truth_clone
        tensor[index,nonneg_labels] = max_clone
        tensor = tensor.view(original_shape)
        return tensor
    else:
        raise TypeError("Rank of tensor must be 2 or 3")

class no_op:
    @staticmethod
    def add_scalar(*args, **kwargs):
        pass

def move_to_device(batch, device):
    r"""Puts each data field to the device"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch,(list,tuple)):
        return tuple(move_to_device(item,device) for item in batch)
    elif isinstance(batch, abc.Mapping):
        return {key: move_to_device(value,device) for key, value in batch.items()}
    else:
        return batch

def get_outputs_from_batch(batch, device, model_T, model_S, args, no_teacher_forward=False):
    if isinstance(batch, abc.Mapping):
        if 'teacher' in batch and 'student' in batch:
            teacher_batch = batch['teacher']
            student_batch = batch['student']
            teacher_batch = move_to_device(teacher_batch, device)
            #teacher outputs
            if no_teacher_forward is True:
                results_T = None
            else:
                if 'teacher_cache' in batch:
                    results_T = move_to_device(batch['teacher_cache'],device)
                else:
                    with torch.no_grad():
                        results_T = auto_forward(model_T,teacher_batch,args)
            #student outputs
            student_batch = move_to_device(student_batch, device)
            if isinstance(student_batch, abc.Mapping):
                results_S = model_S(**student_batch, **args)
            else:
                results_S = model_S(*student_batch, **args)
        else:
            batch = move_to_device(batch,device)
            if no_teacher_forward is True:
                results_T = None
            else:
                with torch.no_grad():
                    results_T = auto_forward(model_T,batch,args)
            results_S = model_S(**batch, **args)
            teacher_batch = student_batch = batch
    else:
        batch = move_to_device(batch,device)
        if no_teacher_forward is True:
            results_T = None
        else:
            with torch.no_grad():
                results_T = auto_forward(model_T,batch,args)
        results_S = model_S(*batch, **args)
        teacher_batch = student_batch = batch
    
    return (teacher_batch,results_T), (student_batch,results_S)

def auto_forward(model,batch,args):
    if isinstance(batch, abc.Mapping):
        if isinstance(model,(list,tuple)):
            results = [v(**batch, **args) for v in model]
        elif isinstance(model,dict):
            results = {k:v(**batch, **args) for k,v in model.items()}
        else:
            results = model(**batch, **args)
    else:
        if isinstance(model,(list,tuple)):
            results = [v(*batch, **args) for v in model]
        elif isinstance(model,dict):
            results = {k:v(*batch, **args) for k,v in model.items()}
        else:
            results = model(*batch, **args)
    return results
