from .distiller_utils import *
from .distiller_basic import BasicDistiller

class MultiTeacherDistiller(BasicDistiller):
    """
    Distills multiple teacher models (of the same tasks) into a student model. **It doesn't support intermediate feature matching**.

    Args:
        train_config (:class:`TrainingConfig`): training configuration.
        distill_config (:class:`DistillationConfig`): distillation configuration.
        model_T (List[torch.nn.Module]): list of teacher models.
        model_S (torch.nn.Module): student model.
        adaptor_T (Callable): teacher model's adaptor.
        adaptor_S (Callable): student model's adaptor.

    The roles of `adaptor_T` and `adaptor_S` are explained in :py:func:`adaptor`.
    """

    def __init__(self, train_config,
                 distill_config,
                 model_T,
                 model_S,
                 adaptor_T,
                 adaptor_S):
        super(MultiTeacherDistiller, self).__init__(
            train_config, distill_config,
            model_T, model_S,
            adaptor_T, adaptor_S)
        if hasattr(self.adaptor_T,'__iter__'):
            assert len(self.adaptor_T)==len(self.model_T)

    def train_on_batch(self, batch, args):
        if self.d_config.is_caching_logits is False:
            (teacher_batch, results_T), (student_batch, results_S) = get_outputs_from_batch(batch, self.t_config.device, self.model_T, self.model_S, args)

            if hasattr(self.adaptor_T,'__iter__'):
                results_T = [post_adaptor(adpt_t(teacher_batch,results_t)) for results_t,adpt_t in zip(results_T,self.adaptor_T)]
            else:
                results_T = [post_adaptor(self.adaptor_T(teacher_batch,results_t)) for results_t in results_T]
            results_S = post_adaptor(self.adaptor_S(student_batch,results_S))
        else:
            batch, cached_logits = batch
            _, (student_batch, results_S) = get_outputs_from_batch(batch, self.t_config.device, self.model_T, self.model_S, args, no_teacher_forward=True)
            results_S = post_adaptor(self.adaptor_S(student_batch,results_S))
            results_T = [{'logits': [lo.to(self.t_config.device) for lo in logits]} for logits in cached_logits]
            if 'logits_mask' in results_S:
                results_T[0]['logits_mask'] = results_S['logits_mask']
    

        logits_list_T = [results_t['logits'] for results_t in results_T]  # list of tensor
        logits_list_S = results_S['logits']  # list of tensor
        total_loss  = 0
        losses_dict = dict()
        total_kd_loss = 0

        if 'logits_mask' in results_S:
            masks_list_S = results_S['logits_mask']
            logits_list_S = select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
        if 'logits_mask' in results_T[0]:
            masks_list_T = results_T[0]['logits_mask']
            logits_list_T = [select_logits_with_mask(logits_list_t,masks_list_T)
                             for logits_list_t in logits_list_T] #(mask_sum, num_of_class)

        if self.d_config.probability_shift is True:
            labels_list = results_S['labels']
            for l_T, l_S, labels in zip(zip(*logits_list_T),logits_list_S,labels_list):
                mean_l_T = sum(l_T)/len(l_T)
                mean_l_T = probability_shift_(mean_l_T, labels)
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(l_S, mean_l_T, self.d_config.temperature)
                else:
                    temperature = self.d_config.temperature
                total_kd_loss += self.kd_loss(l_S, mean_l_T, temperature)
        else:
            for l_T, l_S in zip(zip(*logits_list_T),logits_list_S):
                mean_l_T = sum(l_T)/len(l_T)
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(l_S, mean_l_T, self.d_config.temperature)
                else:
                    temperature = self.d_config.temperature
                total_kd_loss += self.kd_loss(l_S, mean_l_T, temperature)
        total_loss += total_kd_loss * self.d_config.kd_loss_weight
        losses_dict['unweighted_kd_loss'] = total_kd_loss

        if 'losses' in results_S:
            total_hl_loss = 0
            for loss in results_S['losses']:
                # in case of multi-GPU
                total_hl_loss += loss.mean() 
            total_loss += total_hl_loss * self.d_config.hard_label_weight
            losses_dict['unweighted_hard_label_loss'] = total_hl_loss

        return total_loss, losses_dict

    def cache_logits(self, batch, args, batch_postprocessor):
        if batch_postprocessor is not None:
            batch = batch_postprocessor(batch)

        if type(batch) is dict:
            new_batch = {}
            for k,v in batch.items():
                if type(v) is torch.Tensor:
                    new_batch[k] = v.to(self.t_config.device)
                else:
                    new_batch[k] = v
            with torch.no_grad():
                results_T = [model_t(**new_batch, **args) for model_t in self.model_T]
        else:
            new_batch = tuple(item.to(self.t_config.device) if type(item) is torch.Tensor else item for item in batch)
            with torch.no_grad():
                results_T = [model_t(*new_batch, **args) for model_t in self.model_T]

        if hasattr(self.adaptor_T,'__iter__'):
            results_T = [post_adaptor(adpt_t(batch,results_t)) for results_t,adpt_t in zip(results_T,self.adaptor_T)]
        else:
            results_T = [post_adaptor(self.adaptor_T(batch,results_t)) for results_t in results_T]

        self.logits_cache.append([batch, [[logits.to('cpu') for logits in results_t['logits']] for results_t in results_T]])