from .distiller_utils import *
from .distiller_general import GeneralDistiller

class MultiTaskDistiller(GeneralDistiller):
    """
    distills multiple teacher models (of different tasks) into a single student. **It supports intermediate feature matching since 0.2.1**.

    Args:
        train_config (:class:`TrainingConfig`): training configuration.
        distill_config (:class:`DistillationConfig`): distillation configuration.
        model_T (dict): dict of teacher models: {task1:model1, task2:model2, .... }. Keys are tasknames.
        model_S (torch.nn.Module): student model.
        adaptor_T (dict): dict of teacher adaptors: {task1:adpt1, task2:adpt2, .... }. Keys are tasknames.
        adaptor_S (dict): dict of student adaptors: {task1:adpt1, task2:adpt2, .... }. Keys are tasknames.

    """

    def __init__(self, train_config,
                 distill_config,
                 model_T,
                 model_S,
                 adaptor_T,
                 adaptor_S):

        super(MultiTaskDistiller, self).__init__(
            train_config, distill_config,
            model_T, model_S,
            adaptor_T, adaptor_S)
        if hasattr(self.adaptor_T,'__iter__'):
            assert len(self.adaptor_T)==len(self.model_T)==len(self.adaptor_S)
        #assert (self.d_config.kd_loss_weight_scheduler is None) and (self.d_config.hard_label_weight_scheduler is None),\
        #        "MultiTaskDistiller does not support WEIGHT_SCHEDULER in the current version."

        self.d_config.is_caching_logits = False

    def train(self, optimizer, dataloaders, num_steps, scheduler_class=None, scheduler_args=None, scheduler=None, max_grad_norm = -1.0, tau=1, callback=None, batch_postprocessors=None, **args):
        """
        trains the student model.

        Args:
            optimizer: optimizer.
            dataloaders (dict): dict of dataset iterator. Keys are tasknames, values are corresponding dataloaders.
            num_steps (int): number of training steps.
            scheduler_class (class): the class of the scheduler to be constructed.
            scheduler_args (dict): arguments (excluding `optimizer`) passed to the `scheduler_class` to construct the scheduler object.
            scheduler (deprecated): used to adjust learning rate, optional, can be None, is deprecated in favor of `scheduler_class` and `scheduler_args`.
            max_grad_norm (float): Maximum norm for the gradients (-1 means no clipping). Default: -1.0
            tau (float): the probability of sampling an example from task `d` is proportional to \|d\|^{tau}, where \|d\| is the size of `d`'s training set. If the size of any dataset is unknown, ignores tau and samples examples unifromly from each dataset.
            callback (Callable): function called after each epoch, can be None. It is called as ``callback(model=self.model_S, step = global_step)``. It can be used to do evaluation of the model at each checkpoint.
            batch_postprocessors (dict): a dict of batch_postprocessors. Keys are tasknames, values are corresponding batch_postprocessors. Each batch_postprocessor should take a batch and return a batch.
            **args: additional arguments fed to the model.
        """
        optimizer, scheduler, tqdm_disable = self.initialize_training(optimizer, scheduler_class, scheduler_args, scheduler)

        total_global_steps = num_steps
        ckpt_steps = int(self.t_config.ckpt_steps)
        num_steps = int(num_steps)
        print_every = ckpt_steps // self.print_freq
        if print_every == 0:
            print_every = ckpt_steps
        checkpoints = [ i * ckpt_steps for i in range(1,num_steps//ckpt_steps+1)] # + [total_global_steps]
        if checkpoints[-1] != total_global_steps:
            checkpoints.append(total_global_steps)
        logger.info(f"Total training steps: {total_global_steps}")
        logger.info(f"Checkpoints(step): {checkpoints}")

        dataiters = {k:cycle(v) for k,v in dataloaders.items()}
        if all(hasattr(v,'__len__') for v in dataloaders.values()):
            dataloader_sizes = {k:len(v) for k,v in dataloaders.items()}
            total_size = sum(v for k,v in dataloader_sizes.items())//self.t_config.gradient_accumulation_steps
            logger.info(f"Total size of all datasets (in number of batch_size):{total_size}")
            Z = sum(pow(v,tau) for v in dataloader_sizes.values())
            tasknames, sampling_weights = zip(*((k,pow(v,tau)/Z) for k,v in dataloader_sizes.items()))
        else:
            logger.info("The size of some datasets are unknown, so tau=1")
            tasknames = tuple(dataloaders.keys())
            sampling_weights = None


        global_step = 0
        writer_step = 0
        optimizer.zero_grad()
        while global_step < num_steps:
            global_step += 1
            for _ in range(self.t_config.gradient_accumulation_steps):
                #sampling taskname
                taskname = np.random.choice(tasknames,p=sampling_weights)
                dataiter = dataiters[taskname]
                batch = next(dataiter)
                if batch_postprocessors is not None:
                    batch = batch_postprocessors[taskname](batch)
                batch_taskname = (batch, taskname)
                total_loss, losses_dict = self.train_on_batch(batch_taskname, args)

                self.write_loss(total_loss,writer_step,losses_dict)
                writer_step += 1

                total_loss /= self.t_config.gradient_accumulation_steps
                if self.t_config.fp16:
                    with amp.scale_loss(total_loss,optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

            if max_grad_norm > 0:
                if self.t_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model_S.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            if self.d_config.kd_loss_weight_scheduler is not None:
                self.d_config.kd_loss_weight = \
                    self.d_config.kd_loss_weight_scheduler(global_step/total_global_steps)
            if self.d_config.hard_label_weight_scheduler is not None:
                self.d_config.hard_label_weight = \
                    self.d_config.hard_label_weight_scheduler(global_step/total_global_steps)

            if (global_step) % print_every == 0:
                logger.info(f"Global step: {global_step}/{num_steps}")
            if (global_step % ckpt_steps == 0) or global_step==total_global_steps:
                self.save_and_callback(global_step, global_step-1, 0, callback)
        logger.info("Training finished")

    def train_on_batch(self, batch_taskname, args) -> torch.Tensor:
        batch, taskname = batch_taskname
        model_T = self.model_T[taskname]
        adaptor_T = self.adaptor_T[taskname]
        adaptor_S = self.adaptor_S[taskname]

        (teacher_batch, results_T), (student_batch, results_S) = get_outputs_from_batch(batch, self.t_config.device, model_T, self.model_S, args)

        results_T = post_adaptor(adaptor_T(teacher_batch,results_T))
        results_S = post_adaptor(adaptor_S(student_batch,results_S))

        total_loss, losses_dict = self.compute_loss(results_S=results_S, results_T=results_T)

        return total_loss, losses_dict