# coding=utf-8
# Copyright 2022 The OFA-Sys Team. All rights reserved.
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
"""Tokenization classes for OFA."""
from transformers.utils import logging
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
from .tokenization_ofa import OFATokenizer
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import os


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "OFA-Sys/OFA-tiny": "https://huggingface.co/OFA-Sys/OFA-tiny/blob/main/vocab.json",
        "OFA-Sys/OFA-medium": "https://huggingface.co/OFA-Sys/OFA-medium/blob/main/vocab.json",
        "OFA-Sys/OFA-base": "https://huggingface.co/OFA-Sys/OFA-base/blob/main/vocab.json",
        "OFA-Sys/OFA-large": "https://huggingface.co/OFA-Sys/OFA-large/blob/main/vocab.json",
    },
    "merges_file": {
        "OFA-Sys/OFA-tiny": "https://huggingface.co/OFA-Sys/OFA-tiny/blob/main/merges.txt",
        "OFA-Sys/OFA-medium": "https://huggingface.co/OFA-Sys/OFA-medium/blob/main/merges.txt",
        "OFA-Sys/OFA-base": "https://huggingface.co/OFA-Sys/OFA-base/blob/main/merges.txt",
        "OFA-Sys/OFA-large": "https://huggingface.co/OFA-Sys/OFA-large/blob/main/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "OFA-Sys/OFA-tiny": 1024,
    "OFA-Sys/OFA-medium": 1024,
    "OFA-Sys/OFA-base": 1024,
    "OFA-Sys/OFA-large": 1024,
}


class OFATokenizerFast(BartTokenizerFast):
    r"""
    Construct a "fast" OFA tokenizer (backed by HuggingFace's *tokenizers* library).

    [`~OFATokenizerFast`] is identical to [`BartTokenizerFast`] and runs end-to-end tokenization: punctuation splitting
    and wordpiece.

    Refer to superclass [`BartTokenizerFast`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = OFATokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        tokenizer.add_tokens(["<code_{}>".format(i) for i in range(8192)])
        tokenizer.add_tokens(["<bin_{}>".format(i) for i in range(1000)])
        return tokenizer
