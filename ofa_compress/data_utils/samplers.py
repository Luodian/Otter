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
"""batch samplers that work with either random or sequential data samplers"""

import torch
from torch.utils import data


class RandomSampler(data.sampler.Sampler):
    r"""
    Based off of pytorch RandomSampler and DistributedSampler. Essentially a RandomSampler,
    but this class lets the user set an epoch like DistributedSampler
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """
    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.epoch = -1

        if self._num_samples is not None and replacement is False:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(
                                 self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        g = torch.Generator()
        if self.epoch >= 0:
            g.manual_seed(self.epoch)
        if self.replacement:
            return iter(
                torch.randint(high=n,
                              size=(self.num_samples, ),
                              dtype=torch.int64,
                              generator=g).tolist())
        return iter(torch.randperm(n, generator=g).tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSequentialSampler(data.sampler.Sampler):
    def __init__(self,
                 num_samples,
                 train_iters,
                 batch_size,
                 rank=-1,
                 world_size=2):
        super().__init__(num_samples)
        if rank == -1:
            rank = 0
            world_size = 1
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.start_iter = 0
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.batch_bias = [
            i * (num_samples // batch_size) for i in range(batch_size)
        ]

    def __iter__(self):
        for idx in range(self.start_iter, self.train_iters * 10):
            batch = [(idx + bias) % self.num_samples
                     for bias in self.batch_bias]
            tbatch = self._batch(batch)
            yield tbatch

    def __len__(self):
        return self.train_iters

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        start = self.rank * self.batch_size // self.world_size
        end = (self.rank + 1) * self.batch_size // self.world_size
        return batch[start:end]


class TransposedSampler(data.sampler.Sampler):
    """
    Instead of performing sequential sampling, samples array in a transposed fashion given the
    batch size to sampled. Instead of generating the following indices for a batch size of 2
        1 3 5
        2 4 6
    It will generate
        1 2 3
        4 5 6
    """
    def __init__(self, data_source, batch_size, data_sampler=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.len_ds = len(data_source)
        self.strat_width = self.len_ds // batch_size
        #self.strat_width = math.ceil(self.len_ds/batch_size)
        self.data_sampler = data_sampler
        self.wrap_around = 0

    def transpose_helper(self, x):
        """computes index corrseponding to transpose of index x"""
        return ((x % self.batch_size) * self.strat_width +
                (x // self.batch_size)) % self.len_ds
        x += self.wrap_around
        return ((x % self.batch_size) * self.strat_width +
                (x // self.batch_size)) % self.len_ds

    def __iter__(self):
        if self.data_sampler is None:
            return iter(map(self.transpose_helper, range(len(self))))
        return iter(map(self.transpose_helper, iter(self.data_sampler)))

    def __len__(self):
        #return self.len_ds
        return self.strat_width * self.batch_size


class DistributedBatchSampler(data.sampler.BatchSampler):
    """
    similar to normal implementation of distributed sampler, except implementation is at the
    batch sampler level, instead of just the sampler level. This allows wrapping of arbitrary
    data samplers (sequential, random, WeightedRandomSampler, etc.) with this batch sampler.
    """
    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last,
                 rank=-1,
                 world_size=2,
                 wrap_last=False,
                 gradient_accumulation_steps=None):
        super(DistributedBatchSampler, self).__init__(sampler, batch_size,
                                                      drop_last)
        if rank == -1:
            assert False, 'should not be here'
        self.rank = rank
        self.world_size = world_size
        self.sampler.wrap_around = 0
        self.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0
        self.effective_batch_size = batch_size if gradient_accumulation_steps is None else batch_size * gradient_accumulation_steps

    def __iter__(self):
        batch = []
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            if len(batch) == self.batch_size:
                tbatch = self._batch(batch)
                if i >= self.start_iter * self.effective_batch_size:
                    yield tbatch
                    self.start_iter = 0
                i += len(batch)
                batch = []
        batch_len = len(batch)
        if batch_len > 0 and not self.drop_last:
            if self.wrap_last:
                self.sampler.wrap_around -= (self.batch_size)
                self.wrap_around += (len(batch))
                self.wrap_around %= self.batch_size
                if isinstance(self.sampler, TransposedSampler):
                    for i, idx in enumerate(
                            self.data_iterator(self.sampler,
                                               wrap_around=True)):
                        if i == 0:
                            continue
                        batch.append(idx)
                        if len(batch) == self.batch_size:
                            break
            yield self._batch(batch)
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around % self.batch_size:
                continue
            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        start = self.rank * self.batch_size // self.world_size
        end = (self.rank + 1) * self.batch_size // self.world_size
        return batch[start:end]

    def set_epoch(self, epoch):
        self.epoch = epoch
