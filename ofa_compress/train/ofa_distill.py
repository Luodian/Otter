# from textbrewer.distiller_utils import *
from textbrewer.distiller_general import GeneralDistiller
from .ofa_presets import *
from tqdm import tqdm
from torch import nn
from textbrewer.compatibility import mask_dtype, is_apex_available
import random
from textbrewer.distiller_utils import post_adaptor, select_logits_with_mask, probability_shift_, CustomMatch, move_to_device, auto_forward
from typing import Optional, Dict, Union
from collections import OrderedDict

has_apex = is_apex_available()
if has_apex:
    from apex import amp

import logging
logger = logging.getLogger("OFADistillation")

class OFADistiller(GeneralDistiller):
    """
    Supports intermediate features matching. **Recommended for single-teacher single-task distillation**.

    Args:
        train_config (:class:`TrainingConfig`): training configuration.
        distill_config (:class:`DistillationConfig`): distillation configuration.
        model_T (:class:`torch.nn.Module`): teacher model.
        model_S (:class:`torch.nn.Module`): student model.
        adaptor_T (Callable): teacher model's adaptor.
        adaptor_S (Callable): student model's adaptor.
        custom_matches (list): supports more flexible user-defined matches (testing).

    The roles of `adaptor_T` and `adaptor_S` are explained in :py:func:`adaptor`.

    """
    def __init__(self,
                 train_config,
                 distill_config,
                 model_T,
                 model_S,
                 adaptor_T,
                 adaptor_S,
                 custom_matches: Optional[List[CustomMatch]] = None):
        # custom_matches=[{'module_T': module_T, 'module_S':module_S,
        #                 'loss': loss, 'weight': weight},...]
        super(GeneralDistiller,
              self).__init__(train_config, distill_config, model_T, model_S,
                             adaptor_T, adaptor_S)

        self.projs = []
        self.projs_group = []
        for im in self.d_config.intermediate_matches:
            if im.proj is not None:
                projection = im.proj[0]
                dim_in = im.proj[1]
                dim_out = im.proj[2]
                self.projs_group.append(im.proj[3])
                self.projs.append(PROJ_MAP[projection](dim_in, dim_out))
                self.projs[-1].to(self.t_config.device)
            else:
                self.projs.append(None)
                self.projs_group.append(None)

        self.has_custom_matches = False
        if custom_matches:
            self.handles_T = []
            self.handles_S = []
            self.custom_matches_cache = {
                'hook_outputs_T': [],
                'hook_outputs_S': [],
                'match_proj_funcs': [],
                'match_weights': [],
                'match_losses': [],
                'match_proj_groups': []
            }
            for match in custom_matches:
                self.add_match(match)
            self.has_custom_matches = True

        self.d_config.is_caching_logits = False

    def initialize_training(self, optimizer, scheduler_class, scheduler_args,
                            scheduler):
        # update optimizer for projection layer (used in GeneralDistiller)
        if hasattr(self, 'projs'):
            for proj, proj_group in zip(self.projs, self.projs_group):
                if proj is not None:
                    assert isinstance(proj, nn.Module)
                    optimizer.add_param_group({
                        **{
                            'params': proj.parameters()
                        },
                        **proj_group
                    })

        if hasattr(self, 'has_custom_matches') and self.has_custom_matches:
            for proj_func, proj_group in zip(
                    self.custom_matches_cache['match_proj_funcs'],
                    self.custom_matches_cache['match_proj_groups']):
                if isinstance(proj_func, nn.Module):
                    optimizer.add_param_group({
                        **{
                            'params': proj_func.parameters()
                        },
                        **proj_group
                    })

        logger.debug("Optimizer param group: ")
        logger.debug(
            f"{[[s.shape for s in g['params']] for g in optimizer.param_groups]}"
        )

        # update scheduler
        if scheduler_class is not None:
            # overwrite scheduler
            scheduler = scheduler_class(**{'optimizer': optimizer},
                                        **scheduler_args)

        if self.t_config.fp16:
            if not has_apex:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            if isinstance(self.model_T, (list, tuple)):
                models = [self.model_S] + list(self.model_T)
                models, optimizer = amp.initialize(
                    models, optimizer, opt_level=self.t_config.fp16_opt_level)
                self.model_S = models[0]
                self.model_T = models[1:]
            elif isinstance(self.model_T, dict):
                tasknames, model_Ts = zip(*self.model_T.items())
                models = [self.model_S] + list(model_Ts)
                models, optimizer = amp.initialize(
                    models, optimizer, opt_level=self.t_config.fp16_opt_level)
                self.model_S = models[0]
                self.model_T = dict(zip(tasknames, models[1:]))
            else:
                (self.model_S, self.model_T), optimizer = amp.initialize(
                    [self.model_S, self.model_T],
                    optimizer,
                    opt_level=self.t_config.fp16_opt_level)
        if self.local_rank != -1:
            self.model_S = torch.nn.parallel.DistributedDataParallel(
                self.model_S,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
            if self.model_T:
                if isinstance(self.model_T, (list, tuple)):
                    self.model_T = [
                        torch.nn.parallel.DistributedDataParallel(
                            model_t,
                            device_ids=[self.local_rank],
                            output_device=self.local_rank,
                            find_unused_parameters=True,
                            broadcast_buffers=False)
                        for model_t in self.model_T
                    ]
                elif isinstance(self.model_T, dict):
                    self.model_T = {
                        k: torch.nn.parallel.DistributedDataParallel(
                            v,
                            device_ids=[self.local_rank],
                            output_device=self.local_rank,
                            find_unused_parameters=True,
                            broadcast_buffers=False)
                        for k, v in self.model_T.items()
                    }
                else:
                    self.model_T = torch.nn.parallel.DistributedDataParallel(
                        self.model_T,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=True,
                        broadcast_buffers=False)
            if hasattr(self, 'projs'):
                for i, proj in enumerate(self.projs):
                    if proj is not None:
                        assert isinstance(proj, nn.Module)
                        self.projs[
                            i] = torch.nn.parallel.DistributedDataParallel(
                            proj,
                            device_ids=[self.local_rank],
                            output_device=self.local_rank,
                            find_unused_parameters=True,
                            broadcast_buffers=False
                        )
        elif self.t_config.data_parallel:
            self.model_S = torch.nn.DataParallel(self.model_S)
            if self.model_T:
                if isinstance(self.model_T, (list, tuple)):
                    self.model_T = [
                        torch.nn.DataParallel(model_t) for model_t in self.model_T
                    ]
                elif isinstance(self.model_T, dict):
                    self.model_T = {
                        k: torch.nn.DataParallel(v)
                        for k, v in self.model_T.items()
                    }
                else:
                    self.model_T = torch.nn.DataParallel(self.model_T)
        tqdm_disable = None if self.rank == 0 else True
        return optimizer, scheduler, tqdm_disable

    def train_with_num_steps(self, optimizer, scheduler, tqdm_disable,
                             dataloader, max_grad_norm, num_steps, callback,
                             batch_postprocessor, **args):
        if self.d_config.is_caching_logits is True:
            raise AssertionError(
                "You cannot set is_caching_logits to True with num_steps not None!"
            )
        total_global_steps = num_steps
        ckpt_steps = int(self.t_config.ckpt_steps)
        num_steps = int(num_steps)
        print_every = ckpt_steps // self.print_freq
        if print_every == 0:
            print_every = ckpt_steps
        checkpoints = [
                          i * ckpt_steps for i in range(1, num_steps // ckpt_steps + 1)
                      ] + [total_global_steps]
        logger.info(f"Total training steps: {total_global_steps}")
        logger.info(f"Checkpoints(step): {checkpoints}")

        global_step = 0
        writer_step = 0
        for step, batch in tqdm(enumerate(cycle(dataloader)),
                                disable=tqdm_disable):
            if batch_postprocessor is not None:
                batch = batch_postprocessor(batch)
            total_loss, losses_dict = self.train_on_batch(batch, args)

            self.write_loss(total_loss, writer_step, losses_dict)
            writer_step += 1

            total_loss /= self.t_config.gradient_accumulation_steps
            if self.t_config.fp16:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            if (step + 1) % self.t_config.gradient_accumulation_steps == 0:
                if max_grad_norm > 0:
                    if self.t_config.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model_S.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if self.d_config.kd_loss_weight_scheduler is not None:
                    self.d_config.kd_loss_weight = \
                        self.d_config.kd_loss_weight_scheduler(global_step / total_global_steps)
                if self.d_config.hard_label_weight_scheduler is not None:
                    self.d_config.hard_label_weight = \
                        self.d_config.hard_label_weight_scheduler(global_step / total_global_steps)

                if (global_step) % print_every == 0:
                    logger.info(
                        f"Global step: {global_step}, epoch step:{step + 1}")
                if (global_step % ckpt_steps
                    == 0) or global_step == total_global_steps:
                    self.save_and_callback(global_step, step, 0, callback)
                    loss_dict_str = ", ".join(
                        [f"{k}: {v.item()}" for k, v in losses_dict.items()])
                    logger.info(f"Loss dict: {loss_dict_str}")
            if global_step >= total_global_steps:
                logger.info("Training finished")
                return

    def train_with_num_epochs(self, optimizer, scheduler, tqdm_disable,
                              dataloader, max_grad_norm, num_epochs, callback,
                              batch_postprocessor, **args):

        if isinstance(dataloader, list):
            train_steps_per_epoch = sum([len(
                dataloader[i]) for i in range(len(dataloader))]) // len(
                dataloader) // self.t_config.gradient_accumulation_steps
        else:
            train_steps_per_epoch = len(
                dataloader) // self.t_config.gradient_accumulation_steps
        total_global_steps = train_steps_per_epoch * num_epochs
        print_every = train_steps_per_epoch // self.print_freq
        print_every = 100
        if print_every == 0:
            print_every = train_steps_per_epoch
        checkpoints = [
            int(train_steps_per_epoch * ci / self.t_config.ckpt_frequency)
            for ci in range(self.t_config.ckpt_frequency)
        ]
        logger.info(f"Training steps per epoch: {train_steps_per_epoch}")
        logger.info(f"Checkpoints(step): {checkpoints}")

        global_step = 0
        writer_step = 0

        if self.d_config.is_caching_logits is True:
            logger.info(f"Caching batches and teacher's logits...")
            for step, batch in tqdm(enumerate(dataloader),
                                    disable=tqdm_disable):
                self.cache_logits(batch, args, batch_postprocessor)

        for current_epoch in tqdm(range(int(num_epochs)),
                                  disable=tqdm_disable):
            if isinstance(dataloader, list):
                index = current_epoch % len(dataloader)
                dataloader_single = dataloader[index]
            else:
                dataloader_single = dataloader

            if self.local_rank != -1 and hasattr(dataloader_single, 'sampler'):
                dataloader_single.sampler.set_epoch(
                    current_epoch
                )  # In distributed mode, calling the set_epoch method is needed to make shuffling work;
            logger.info(f"Epoch {current_epoch + 1}")
            optimizer.zero_grad()
            if self.d_config.is_caching_logits:
                random.shuffle(self.logits_cache)
                dataloader_single = self.logits_cache
            logger.info(
                f"Length of current epoch in forward batch: {len(dataloader_single)}  {dataloader_single}")
            for step, batch in tqdm(enumerate(dataloader_single),
                                    disable=tqdm_disable):
                if self.d_config.is_caching_logits is False and batch_postprocessor is not None:
                    batch = batch_postprocessor(batch)
                total_loss, losses_dict = self.train_on_batch(batch, args)

                self.write_loss(total_loss, writer_step, losses_dict)
                writer_step += 1

                total_loss /= self.t_config.gradient_accumulation_steps
                if self.t_config.fp16:
                    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                if (step + 1) % self.t_config.gradient_accumulation_steps == 0:
                    if max_grad_norm > 0:
                        if self.t_config.fp16:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(optimizer), max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model_S.parameters(), max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if self.d_config.kd_loss_weight_scheduler is not None:
                        self.d_config.kd_loss_weight = \
                            self.d_config.kd_loss_weight_scheduler(global_step / total_global_steps)
                    if self.d_config.hard_label_weight_scheduler is not None:
                        self.d_config.hard_label_weight = \
                            self.d_config.hard_label_weight_scheduler(global_step / total_global_steps)

                    if (global_step) % print_every == 0:
                        logger.info(
                            f"Global step: {global_step}, epoch step:{step + 1}")
                        logger.info(f"Loss dict: {losses_dict}")
                        # logger.info(f"Total loss: {total_loss}")
                    if (global_step % train_steps_per_epoch in checkpoints) \
                            and ((
                                         current_epoch + 1) % self.t_config.ckpt_epoch_frequency == 0 or current_epoch + 1 == num_epochs):
                        losses_str_dict = {
                            k: v.item()
                            for k, v in losses_dict.items()
                        }
                        logger.info(f"Loss dict: {losses_str_dict}")
                        self.save_and_callback(global_step, step,
                                               current_epoch, callback)

            logger.info(f"Epoch {current_epoch + 1} finished")

    def save_and_callback(self, global_step, step, epoch, callback):
        if self.has_custom_matches:
            if self.model_T:
                handles_T = self.model_T._forward_hooks
            handles_S = self.model_S._forward_hooks
            self.model_S._forward_hooks = OrderedDict()  # clear hooks
            self.model_T._forward_hooks = OrderedDict()

        super(GeneralDistiller,
              self).save_and_callback(global_step, step, epoch, callback)

        if self.has_custom_matches:
            self.model_S._forward_hooks = handles_S  # restore hooks
            if self.model_T:
                self.model_T._forward_hooks = handles_T

    def train_on_batch(self, batch, args):
        if self.model_T:

            (teacher_batch,
             results_T), (student_batch, results_S) = get_outputs_from_batch(
                 batch, self.t_config.device, self.model_T, self.model_S, args)

            results_T = post_adaptor(self.adaptor_T(teacher_batch, results_T))
            results_S = post_adaptor(self.adaptor_S(student_batch, results_S))
        else:
            (teacher_batch,
             results_T), (student_batch, results_S) = get_outputs_from_batch(
                batch, self.t_config.device, self.model_T, self.model_S,args, no_teacher_forward= True)
            results_S = post_adaptor(self.adaptor_S(student_batch, results_S))

        total_loss, losses_dict = self.compute_loss(results_S, results_T)
        t_loss = 0
        if self.model_T:
            for loss in results_T['losses']:
                # in case of multi-GPU
                t_loss += loss.mean()
                losses_dict["Teacher_loss"] = t_loss
        s_loss = 0
        for loss in results_S['losses']:
            # in case of multi-GPU
            s_loss += loss.mean()
        losses_dict["Student_loss"] = s_loss

        return total_loss, losses_dict

    def compute_loss(self, results_S, results_T):
        losses_dict = dict()
        total_loss = 0
        if 'logits' in results_T and 'logits' in results_S:
            logits_list_T = results_T['logits']  # list of tensor
            logits_list_S = results_S['logits']  # list of tensor
            total_kd_loss = 0
            if 'logits_mask' in results_S:
                masks_list_S = results_S['logits_mask']
                logits_list_S = select_logits_with_mask(
                    logits_list_S, masks_list_S)  #(mask_sum, num_of_class)
            if 'logits_mask' in results_T:
                masks_list_T = results_T['logits_mask']
                logits_list_T = select_logits_with_mask(
                    logits_list_T, masks_list_T)  #(mask_sum, num_of_class)

            if self.d_config.probability_shift is True:
                labels_list = results_S['labels']
                for l_T, l_S, labels in zip(logits_list_T, logits_list_S,
                                            labels_list):
                    l_T = probability_shift_(l_T, labels)
                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(
                            l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature
                    if self.d_config.kd_loss_type.endswith("with_mask"):
                        constraint_masks = None
                        if "constraint_masks" in results_S:
                            constraint_masks = results_S["constraint_masks"]
                        total_kd_loss += self.kd_loss(l_S, l_T, results_S['target'],
                                                      self.d_config.text_preprocessor.pad_token_id, temperature,
                                                      self.d_config.constraint_range, constraint_masks)
                    else:
                        total_kd_loss += self.kd_loss(l_S, l_T, temperature)
            else:
                for l_T, l_S in zip(logits_list_T, logits_list_S):
                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(
                            l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature
                    if self.d_config.kd_loss_type.endswith("with_mask"):
                        constraint_masks = None
                        if "constraint_masks" in results_S:
                            constraint_masks = results_S["constraint_masks"]
                        total_kd_loss += self.kd_loss(l_S, l_T, results_S['target'],
                                                      self.d_config.text_preprocessor.pad_token_id, temperature,
                                                      self.d_config.constraint_range, constraint_masks)
                        # logits_S, logits_T, target, padding_idx, temperature=1, constraint_range=None,constraint_masks=None
                    else:
                        total_kd_loss += self.kd_loss(l_S, l_T, temperature)
            total_loss += total_kd_loss * self.d_config.kd_loss_weight
            losses_dict['unweighted_kd_loss'] = total_kd_loss
        try:
            inters_T = {
                feature: results_T.get(feature, [])
                for feature in FEATURES
            }

            inputs_mask_T = results_T.get('inputs_mask', None)
        except Exception as e:
            pass
        inters_S = {
            feature: results_S.get(feature, [])
            for feature in FEATURES
        }
        inputs_mask_S = results_S.get('inputs_mask', None)
        inputs_target_S = results_S.get('target', None)
        for ith, inter_match in enumerate(self.d_config.intermediate_matches):
            layer_T = inter_match.layer_T
            layer_S = inter_match.layer_S
            feature = inter_match.feature
            loss_type = inter_match.loss
            match_weight = inter_match.weight
            xcoder = inter_match.xcoder
            match_loss = MATCH_LOSS_MAP[loss_type]
            if xcoder == 'decoder':
                inputs_mask_T = inputs_mask_S \
                    = torch.where(inputs_target_S!=0,torch.ones_like(inputs_target_S),torch.zeros_like(inputs_target_S))

            if type(layer_S) is list and type(layer_T) is list:
                inter_S = [inters_S[feature][s] for s in layer_S]
                inter_T = [inters_T[feature][t] for t in layer_T]
                name_S = '-'.join(map(str, layer_S))
                name_T = '-'.join(map(str, layer_T))
                if self.projs[ith]:
                    #inter_T = [self.projs[ith](t) for t in inter_T]
                    inter_S = [self.projs[ith](s) for s in inter_S]
            else:
                inter_S = inters_S[feature][layer_S]
                inter_T = inters_T[feature][layer_T]
                name_S = str(layer_S)
                name_T = str(layer_T)
                if self.projs[ith]:
                    inter_S = self.projs[ith](inter_S)
            intermediate_loss = match_loss(inter_S,
                                           inter_T,
                                           mask=inputs_mask_S)
            total_loss += intermediate_loss * match_weight
            losses_dict[
                f'unweighted_{feature}_{loss_type}_{name_S}_{name_T}'] = intermediate_loss

        if self.has_custom_matches:
            for hook_T, hook_S, match_weight, match_loss, proj_func  in \
                    zip(self.custom_matches_cache['hook_outputs_T'], self.custom_matches_cache['hook_outputs_S'],
                        self.custom_matches_cache['match_weights'], self.custom_matches_cache['match_losses'],
                        self.custom_matches_cache['match_proj_funcs']):
                if proj_func is not None:
                    hook_S = proj_func(hook_S)
                total_loss += match_weight * match_loss(
                    hook_S, hook_T, inputs_mask_S, inputs_mask_T)
            self.custom_matches_cache['hook_outputs_T'] = []
            self.custom_matches_cache['hook_outputs_S'] = []

        if 'losses' in results_S:
            total_hl_loss = 0
            for loss in results_S['losses']:
                # in case of multi-GPU
                total_hl_loss += loss.mean()
            total_loss += total_hl_loss * self.d_config.hard_label_weight
            losses_dict['unweighted_hard_label_loss'] = total_hl_loss
        return total_loss, losses_dict

    def add_match(self, match: CustomMatch):
        if type(match.module_T) is str or type(match.module_S) is str:
            raise NotImplementedError
        else:
            module_T = match.module_T
            module_S = match.module_S
            weight = match.weight
            loss = match.loss
            proj_func = match.proj_func
            proj_group = match.proj_group
        self.add_match_by_module(module_T, module_S, proj_func, proj_group,
                                 weight, loss)

    def add_match_by_module(self, module_T: torch.nn.Module,
                            module_S: torch.nn.Module, proj_func, proj_group,
                            match_weight, match_loss):

        self.handles_T = module_T.register_forward_hook(self._hook_T)
        self.handles_S = module_S.register_forward_hook(self._hook_S)
        self.custom_matches_cache['match_weights'].append(match_weight)
        self.custom_matches_cache['match_losses'].append(match_loss)
        self.custom_matches_cache['match_proj_funcs'].append(proj_func)
        if isinstance(proj_func, nn.Module):
            self.custom_matches_cache['match_proj_funcs'][-1].to(
                self.t_config.device)
        self.custom_matches_cache['match_proj_groups'].append(proj_group)

    def _hook_T(self, module, input, output):
        self.custom_matches_cache['hook_outputs_T'].append(output)

    def _hook_S(self, module, input, output):
        self.custom_matches_cache['hook_outputs_S'].append(output)


def get_outputs_from_batch(batch, device, model_T, model_S, args, no_teacher_forward=False):
    if type(batch) is dict:
        if 'teacher' in batch and 'student' in batch:
            teacher_batch = batch['teacher']
            student_batch = batch['student']
            teacher_batch = move_to_device(teacher_batch, device)
            # teacher outputs
            if no_teacher_forward is True:
                results_T = {}
            else:
                if 'teacher_cache' in batch:
                    results_T = move_to_device(batch['teacher_cache'], device)
                else:
                    with torch.no_grad():
                        results_T = auto_forward(model_T, teacher_batch["net_input"], args)
            # student outputs
            student_batch = move_to_device(student_batch, device)
            if type(student_batch) is dict:
                results_S = model_S(**student_batch, **args)
            else:
                results_S = model_S(*student_batch, **args)
        else:
            batch = move_to_device(batch, device)
            if no_teacher_forward is True:
                results_T = {}
            else:
                with torch.no_grad():
                    results_T = auto_forward(model_T, batch["net_input"], args)
            results_S = model_S(**batch["net_input"], **args)
            teacher_batch = student_batch = batch

    elif isinstance(batch, (list, tuple)):
        teacher_batch, results_T, student_batch, results_S = [], [], [], []
        for i in range(len(batch)):
            batch[i] = move_to_device(batch[i], device)
            if no_teacher_forward is True:
                results_T = {}
            else:
                with torch.no_grad():
                    results_T.append(auto_forward(model_T, batch[i]["net_input"], args))
            results_S.append(model_S(**batch[i]["net_input"], **args))
            teacher_batch.append(batch[i])
            student_batch.append(batch[i])
    else:
        batch = move_to_device(batch, device)
        if no_teacher_forward is True:
            results_T = {}
        else:
            with torch.no_grad():
                results_T = auto_forward(model_T, batch["net_input"], args)
        results_S = model_S(*batch["net_input"], **args)
        teacher_batch = student_batch = batch

    return (teacher_batch, results_T), (student_batch, results_S)