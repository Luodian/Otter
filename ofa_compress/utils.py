# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for logging and serialization"""

import os
import random
import time

import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.parallel.distributed import DistributedDataParallel


SUMMARY_WRITER_DIR_NAME = 'runs'


def from_json_file(path):
    with open(path) as f:
        data = json.loads(f.read())
    return data


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)





class Timers:
    """Group of timers."""
    class Timer:
        """Timer."""
        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, names=None, normalizer=1.0, reset=True):
        """Log a group of timers."""
        if names is None:
            names = self.timers.keys()
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print_rank_0(string)


def report_memory(name):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(torch.cuda.memory_allocated() /
                                        mega_bytes)
    string += ' | max allocated: {}'.format(torch.cuda.max_memory_allocated() /
                                            mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}'.format(torch.cuda.memory_reserved() /
                                         mega_bytes)
    print_rank_0(string)


def get_checkpoint_name(checkpoints_path,
                        iteration,
                        best=False,
                        release=False,
                        zero=False):
    if release:
        d = 'release'
    else:
        d = '{:d}'.format(iteration)
    if zero:
        dp_rank = torch.distributed.get_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    if best:
        d = 'best'
    return os.path.join(checkpoints_path, f'{d}_model_states.pt')


def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_zero_checkpoint(args, iteration, optimizer):
    zero_sd = {
        'iteration': iteration,
        'optimizer_state_dict': optimizer.state_dict()
    }
    zero_checkpoint_name = get_checkpoint_name(args.save, iteration, zero=True)
    ensure_directory_exists(zero_checkpoint_name)
    torch.save(zero_sd, zero_checkpoint_name)
    print('  successfully saved {}'.format(zero_checkpoint_name))


def save_checkpoint(iteration, model, args, bucket=None, best=False):
    """Save a model checkpoint."""
    if bucket is not None:
        raise NotImplementedError
    else:
        # Only rank zer0 of the data parallel writes to the disk.
        if isinstance(model, DistributedDataParallel):
            model = model.module

        if torch.distributed.get_rank() == 0:
            checkpoint_name = get_checkpoint_name(args.output_dir,
                                                  iteration,
                                                  best=best)
            print(
                'global rank {} is saving checkpoint at iteration {:7d} to {}'.
                format(torch.distributed.get_rank(), iteration,
                       checkpoint_name))

            sd = {}
            sd['iteration'] = iteration
            sd['module'] = model.state_dict()
            model.save_pretrained(os.path.join(args.output_dir, "saved_mode"))

            ensure_directory_exists(checkpoint_name)
            # torch.save(sd, checkpoint_name)
            print('Successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    # torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.output_dir)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    # Wait so everyone is done (not necessary)
    # torch.distributed.barrier()


def get_checkpoint_iteration(args):
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(args.load)
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                exit()

    # assert iteration > 0 or release, 'error parsing metadata file {}'.format(
    # tracker_filename)

    return iteration, release, True


def load_model_state_only(model,
                          args,
                          remove_prefix=None,
                          remap_prefix=None,
                          force_remap=False):
    if (args.load.split('/')[-1]).isdigit():
        iteration = eval(args.load.split('/')[-1])
        release = False
        args.load = "/".join(args.load.split('/')[:-1])
    else:
        iteration, release, _ = get_checkpoint_iteration(args)
    checkpoint_name = get_checkpoint_name(args.load, iteration, release)

    # Load the checkpoint.
    sd = torch.load(checkpoint_name, map_location='cpu')

    if isinstance(model, DistributedDataParallel):
        model = model.module
    model_state = sd['module'] if 'module' in sd else sd

    if remove_prefix:
        for load_prefix in remove_prefix:
            keys = list(model_state.keys())
            for k in keys:
                if k.startswith(load_prefix):
                    print('Skip loading %s in the checkpoint.' % k)
                    del model_state[k]

    if remap_prefix:
        for var_prefix, load_prefix in remap_prefix.items():
            keys = list(model_state.keys())
            for k in keys:
                if k.startswith(load_prefix):
                    new_k = k.replace(load_prefix, var_prefix)
                    if new_k in model_state:
                        print('WARN: param %s already in the checkpoint.' %
                              new_k)
                    if (new_k not in model_state) or force_remap:
                        print('Load param %s from %s in the checkpoint.' %
                              (new_k, k))
                        model_state[new_k] = model_state[k]

    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as e:
        print(e)
        print(
            '> strict load failed (see error message above), try non-strict load instead'
        )
        keys = model.load_state_dict(model_state, strict=False)
        print('> non-strict load done')
        print(keys)
    return iteration


def load_checkpoint(model,
                    optimizer,
                    lr_scheduler,
                    args,
                    load_optimizer_states=True,
                    bucket=None,
                    load_module_strict=True):
    """Load a model checkpoint."""
    iteration, release, success = get_checkpoint_iteration(args)

    if not success:
        return 0

    if bucket is not None:
        raise NotImplementedError

    else:

        # Checkpoint.
        checkpoint_name = get_checkpoint_name(args.load, iteration, release)

        if torch.distributed.get_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        # Load the checkpoint.
        sd = torch.load(checkpoint_name, map_location='cpu')

        if isinstance(model, DistributedDataParallel):
            model = model.module

        # Model.
        try:
            model.load_state_dict(sd['module'])
            print_rank_0('load_checkpoint - loaded module')
        except KeyError as e:
            print_rank_0('A metadata file exists but unable to load model '
                         'from checkpoint {}, exiting'.format(checkpoint_name))
            raise e

        # Optimizer.
        if not release and not args.finetune and not args.no_load_optim:
            try:
                if optimizer is not None and load_optimizer_states:
                    optimizer.load_state_dict(sd['optimizer'])
                    print_rank_0('load_checkpoint - loaded optimizer')
                if lr_scheduler is not None:
                    lr_scheduler.load_state_dict(sd['lr_scheduler'])
                    print_rank_0('load_checkpoint - loaded lr_scheduler')
            except KeyError as e:
                print_rank_0(
                    'Unable to load optimizer from checkpoint {}, exiting. '
                    'Specify --no-load-optim or --finetune to prevent '
                    'attempting to load the optimizer '
                    'state.'.format(checkpoint_name))
                raise e

    # Iterations.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = sd['iteration']
            print_rank_0('load_checkpoint - loaded iteration')
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = sd['total_iters']
                print_rank_0('load_checkpoint - loaded total_iters')
            except KeyError as e:
                print_rank_0(
                    'A metadata file exists but Unable to load iteration '
                    ' from checkpoint {}, exiting'.format(checkpoint_name))
                raise e

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            print_rank_0('load_checkpoint - loaded rng')
        except KeyError as e:
            print_rank_0(
                'Unable to load optimizer from checkpoint {}, exiting. '
                'Specify --no-load-rng or --finetune to prevent '
                'attempting to load the random '
                'state.'.format(checkpoint_name))
            raise e

    if torch.distributed.get_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration


def guarded_roc_auc_score(y_true, y_score):
    num_true = sum(y_true)
    if (num_true == 0) or (num_true >= len(y_true)):
        return -1
    return roc_auc_score(y_true=y_true, y_score=y_score)


class WindowAUC(object):
    def __init__(self, window_size=1024):
        self.window_size = window_size
        self.y_true = []
        self.y_score = []

    def extend(self, y_true, y_score):
        self.y_true.extend(y_true)
        self.y_true = self.y_true[-self.window_size:]
        self.y_score.extend(y_score)
        self.y_score = self.y_score[-self.window_size:]

    def compute_auc(self):
        return guarded_roc_auc_score(self.y_true, self.y_score)


def bert_input_constructor(batch_size, seq_len, tokenizer):
    fake_seq = ""
    for _ in range(seq_len -
                   2):  # ignore the two special tokens [CLS] and [SEP]
        fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * batch_size)
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs
