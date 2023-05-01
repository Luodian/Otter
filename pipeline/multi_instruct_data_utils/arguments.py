# coding=utf-8
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
"""argparser configuration"""

import argparse
import os

import torch


def add_training_args(parser):
    """Training arguments."""

    group = parser.add_argument_group("train", "training configurations")

    group.add_argument(
        "--batch-size", type=int, default=4, help="Data Loader batch size"
    )
    group.add_argument("--micro-batch-size", type=int, default=0)
    group.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="weight decay coefficient for L2 regularization",
    )
    group.add_argument("--clip-grad", type=float, default=1.0, help="gradient clipping")
    group.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="num-epochs<=0 means to loop forever until >=train-iters",
    )
    group.add_argument(
        "--train-iters",
        type=int,
        default=1000000,
        help="total number of iterations to train over all training runs",
    )

    group.add_argument("--seed", type=int, default=1234, help="random seed")

    # Learning rate.
    group.add_argument(
        "--lr-decay-iters",
        type=int,
        default=None,
        help="number of iterations to decay LR over,"
        " If None defaults to `--train-iters`*`--epochs`",
    )
    group.add_argument(
        "--lr-decay-style",
        type=str,
        default="linear",
        choices=["constant", "linear", "cosine", "exponential"],
        help="learning rate decay function",
    )
    group.add_argument("--lr-decay-ratio", type=float, default=0.1)
    group.add_argument("--lr", type=float, default=1.0e-3, help="initial learning rate")
    group.add_argument(
        "--warmup-proportion",
        type=float,
        default=0.01,
        help="percentage of data to warmup on (.01 = 1% of all "
        "training iters). Default 0.01",
    )
    group.add_argument(
        "--adam-eps",
        type=float,
        default=1e-6,
        help="Adam’s epsilon for numerical stability",
    )
    group.add_argument("--lr-end", type=float, default=1e-7, help="end_learning_rate")
    # model checkpointing

    group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save checkpoints to.",
    )

    group.add_argument(
        "--postfix",
        type=str,
        default=None,
        help="Output directory to save checkpoints to.",
    )

    group.add_argument("--oss", action="store_true", help="Save the model to oss.")
    group.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to a directory containing a model checkpoint.",
    )

    # distributed training args
    group.add_argument(
        "--distributed-backend",
        default="nccl",
        help="which backend to use for distributed " "training. One of [gloo, nccl]",
    )

    group.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="local rank passed from distributed launcher",
    )
    group.add_argument("--worker-cnt", type=int, default=1, help="number of workers")
    group.add_argument(
        "--gpus-per-node", type=int, default=4, help="number of gpus per node"
    )
    group.add_argument("--entry", type=str, default="main_distill.py")
    group.add_argument("--fp16", action="store_true", help="Run model in fp16 mode")

    return parser


def add_data_args(parser=None):
    """Train/valid/test data arguments."""
    if parser is None:
        parser = argparse.ArgumentParser()
    group = parser.add_argument_group("data", "data configurations")

    group.add_argument(
        "--model-parallel-size", type=int, default=1, help="size of the model parallel."
    )
    group.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle data. Shuffling is deterministic "
        "based on seed and current epoch.",
    )
    group.add_argument(
        "--local-shuffle",
        action="store_true",
        help="local shuffle data (used in Image Retrieval finetune)",
    )
    group.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=2000,
        help="local shuffle buffer size (used in Image Retrieval finetune)",
    )
    group.add_argument(
        "--train-data",
        nargs="+",
        default=None,
        help="Whitespace separated filenames or corpora names " "for training.",
    )
    group.add_argument(
        "--use-npy-data-loader",
        action="store_true",
        help="Use the numpy data loader. If set, then"
        "train-data-path, val-data-path, and test-data-path"
        "should also be provided.",
    )
    group.add_argument(
        "--train-data-path", type=str, default="", help="path to the training data"
    )
    group.add_argument(
        "--val-data-path", type=str, default="", help="path to the validation data"
    )
    group.add_argument(
        "--test-data-path", type=str, default="", help="path to the test data"
    )
    group.add_argument(
        "--input-data-sizes-file",
        type=str,
        default="sizes.txt",
        help="the filename containing all the shards sizes",
    )

    group.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="""Number of workers to use for dataloading""",
    )
    group.add_argument(
        "--tokenizer-model-type",
        type=str,
        default="bert-base-chinese",
        help="Model type to use for sentencepiece tokenization \
                       (one of ['bpe', 'char', 'unigram', 'word']) or \
                       bert vocab to use for BertWordPieceTokenizer (one of \
                       ['bert-large-uncased', 'bert-large-cased', etc.])",
    )
    group.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer.model",
        help="path used to save/load sentencepiece tokenization " "models",
    )
    group.add_argument(
        "--tokenizer-type",
        type=str,
        default="BertWordPieceTokenizer",
        choices=[
            "CharacterLevelTokenizer",
            "SentencePieceTokenizer",
            "BertWordPieceTokenizer",
            "GPT2BPETokenizer",
            "ChineseSPTokenizer",
        ],
        help="what type of tokenizer to use",
    )
    group.add_argument("--not-pre-tokenize", action="store_true")
    group.add_argument(
        "--cache-dir",
        default=None,
        type=str,
        help="Where to store pre-trained BERT downloads",
    )
    group.add_argument(
        "--use-tfrecords",
        action="store_true",
        help="load `--train-data`, `--valid-data`, "
        "`--test-data` from BERT tf records instead of "
        "normal data pipeline",
    )
    group.add_argument(
        "--seq-length", type=int, default=512, help="Maximum sequence length to process"
    )
    group.add_argument(
        "--prompt-length",
        type=int,
        default=512,
        help="Maximum prompt length to process",
    )
    group.add_argument(
        "--mem-length", type=int, default=0, help="The memory length to preserve"
    )
    group.add_argument(
        "--max-preds-per-seq",
        type=int,
        default=None,
        help="Maximum number of predictions to use per sequence."
        "Defaults to math.ceil(`--seq-length`*.15/10)*10."
        "MUST BE SPECIFIED IF `--use-tfrecords` is True.",
    )
    group.add_argument(
        "--sample-one-document",
        action="store_true",
        help="only sample one document in one sample",
    )
    group.add_argument(
        "--tables", type=str, default="", help="table name (train, valid, test)"
    )
    group.add_argument(
        "--selected-cols", type=str, default="0,1,2,3,4,5,6,7", help="table column name"
    )
    group.add_argument(
        "--num-bins", type=int, default=1000, help="number of quantization bins"
    )
    group.add_argument("--max-image-size", type=int, default=512, help="max image size")
    group.add_argument(
        "--no-text-data", type=bool, default=False, help="no use pure text data"
    )
    group.add_argument(
        "--no-image-data", type=bool, default=False, help="no use pure image data"
    )
    group.add_argument(
        "--text-selected-cols",
        type=str,
        default=None,
        help="pure text table selected cols",
    )
    group.add_argument(
        "--image-selected-cols",
        type=str,
        default=None,
        help="pure image table selected cols",
    )
    group.add_argument(
        "--detection-selected-cols",
        type=str,
        default=None,
        help="detection table selected cols",
    )
    group.add_argument(
        "--neg-sample-dir", type=str, default=None, help="negative sample dir"
    )
    group.add_argument(
        "--max-object-length",
        type=int,
        default=100,
        help="the maximum object sequence length",
    )
    group.add_argument(
        "--code-dict-size", type=int, default=8192, help="code dict size"
    )
    group.add_argument(
        "--code-image-size", type=int, default=128, help="code image size"
    )
    group.add_argument("--pretrain-seed", type=int, default=7, help="pretrain seed")
    group.add_argument(
        "--mask-ratio",
        type=float,
        default=0.3,
        help="fraction of words/subwords that will be masked",
    )
    group.add_argument(
        "--random-ratio",
        type=float,
        default=0.0,
        help="instead of using [MASK], use random token this often",
    )
    group.add_argument(
        "--keep-ratio",
        type=float,
        default=0.0,
        help="instead of using [MASK], keep original token this often",
    )
    group.add_argument(
        "--mask-length",
        type=str,
        default="span-poisson",
        help="mask length to choose ['subword', 'word', 'span-poisson']",
    )
    group.add_argument(
        "--poisson-lambda",
        type=float,
        default=3.0,
        help="randomly shuffle sentences for this proportion of inputs",
    )
    group.add_argument(
        "--replace-length",
        type=int,
        default=1,
        help="when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)",
    )
    group.add_argument("--seq2seq", action="store_true", help="to use seq2seq dataset")
    group.add_argument("--outputs", type=str, default="")
    group.add_argument("--patch-image-size", type=int, default=224)
    group.add_argument("--imagenet-default-mean-and-std", type=bool, default=False)
    group.add_argument(
        "--max-src-length",
        type=int,
        default=128,
        help="the maximum src sequence length",
    )
    group.add_argument(
        "--max-tgt-length",
        type=int,
        default=30,
        help="the maximum target sequence length",
    )
    group.add_argument("--prompt-type", type=str, default=None, help="prompt_type")
    group.add_argument(
        "--add-object", type=bool, default=False, help="add object to encoder"
    )
    group.add_argument(
        "--add-caption", type=bool, default=False, help="add caption to encoder"
    )

    return parser


def add_custom_args(parser):
    group = parser.add_argument_group(
        "Custom arguments diverted from M6-opensource", "configurations"
    )
    group.add_argument(
        "--num_prompts", type=int, default=0, help="enable prompt tune-tuning if > 0"
    )

    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--do-train", action="store_true")
    group.add_argument("--do-eval", action="store_true")
    group.add_argument("--do-predict", action="store_true")
    group.add_argument("--task", type=str, default="mlm")
    group.add_argument("--schedule", type=str, default="cosine")
    group.add_argument(
        "--ckpt-frequency",
        type=int,
        default=-1,
        help="stores model weights ckpt_frequency times every epoch",
    )
    group.add_argument(
        "--ckpt-epoch-frequency",
        type=int,
        default=1,
        help="stores model weights every ckpt_epoch_frequency times",
    )
    group.add_argument(
        "--metric", type=str, default="accuracy", help="metric to save the best ckpt"
    )
    group.add_argument(
        "--best-score", type=float, default=-1.0, help="save the best score"
    )
    group.add_argument(
        "--best-step",
        type=int,
        default=-1,
        help="The step when model achieve the best score",
    )
    group.add_argument(
        "--generator-version",
        type=str,
        default="fairseq",
        help="The version of generator",
    )
    group.add_argument("--debug-generate", action="store_true")
    group.add_argument(
        "--keep-last-ckpt-num", type=int, default=15, help="The num of ckpts to keep"
    )
    group.add_argument(
        "--evaluate-idx", type=int, default=0, help="The num of evaluate"
    )
    return parser


def add_distill_args(parser):
    group = parser.add_argument_group("Distillation arguments", "configurations")

    group.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Typically range from 1 to 10. Better performance when larger than 1",
    )
    group.add_argument(
        "--temperature-scheduler",
        type=str,
        default="none",
        help="none, constant, flsw or cwsm",
    )
    group.add_argument(
        "--temperature-beta",
        type=float,
        default=1,
        help="used when temp-scheduler is flsw or cwsm",
    )
    group.add_argument(
        "--temperature-gamma",
        type=float,
        default=2,
        help="used when temp-scheduler is cwsm",
    )

    group.add_argument("--hard-label-weight", type=float, default=1)
    group.add_argument(
        "--hard-label-weight-scheduler",
        type=str,
        default="none",
        help="none, linear_decay, linear_growth",
    )

    group.add_argument("--kd-loss-type", type=str, default="ce", help="ce or mse")
    group.add_argument(
        "--kd-loss-weight",
        type=float,
        default=1,
        help="used when temp-scheduler is cwsm",
    )
    group.add_argument(
        "--kd-loss-weight-scheduler",
        type=str,
        default="none",
        help="none, linear_decay, linear_growth",
    )

    group.add_argument(
        "--probability-shift",
        type=bool,
        default=False,
        help="switch the ground-truth label logit and the largest logit predicted by the teacher. Need labels returned by the adaptor",
    )

    group.add_argument(
        "--intermediate-matches",
        type=str,
        default="",
        help="The inetrmediate matches name in `matches.py`",
    )

    group.add_argument("--is-caching-logits", type=bool, default=False)

    return parser


def add_model_args(parser):
    group = parser.add_argument_group("Load&Save arguments", "configurations")
    group.add_argument(
        "--load-teacher-model",
        type=str,
        default="",
        help="Load the teacher model in volume",
    )
    group.add_argument(
        "--init-method",
        type=str,
        default="load_pretrain",
        help="The method to initialize the student model.",
    )
    group.add_argument(
        "--load-student-model",
        type=str,
        default="",
        help="Load the student model in volume",
    )
    group.add_argument(
        "--student-model-config",
        type=str,
        default="base",
        help="The config name of student model",
    )

    group.add_argument(
        "--scst", type=bool, default=False, help="Self-critical sequence training"
    )
    return parser


def add_criterions_args(parser):
    group = parser.add_argument_group("Criterions arguments", "configurations")
    group.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="epsilon for label smoothing, 0 means no label smoothing",
    )
    group.add_argument(
        "--report-accuracy", type=bool, default=False, help="report accuracy metric"
    )
    group.add_argument(
        "--ignore-prefix-size", type=int, default=0, help="Ignore first N tokens"
    )
    group.add_argument(
        "--ignore-eos", type=bool, default=False, help="Ignore eos token"
    )
    # group.add_argument("--sentence_avg",
    #                    type=bool)
    group.add_argument(
        "--drop-worst-ratio",
        type=float,
        default=0.0,
        help="ratio for discarding bad samples",
    )
    group.add_argument(
        "--drop-worst-after",
        type=int,
        default=0,
        help="steps for discarding bad samples",
    )
    group.add_argument("--use-rdrop", type=bool, default=False, help="use R-Drop")
    group.add_argument("--reg-alpha", type=float, default=1.0, help="weight for R-Drop")
    group.add_argument(
        "--sample-patch-num", type=int, default=196, help="sample patches for v1"
    )
    group.add_argument(
        "--constraint-range", type=str, default=None, help="constraint range"
    )
    group.add_argument(
        "--sentence-avg",
        type=bool,
        default=False,
        help="normalize gradients by the number of sentences in a batch",
    )
    group.add_argument(
        "--eval-cider-cached-tokens",
        type=str,
        default=None,
        help="path to cached cPickle file used to calculate CIDEr scores",
    )
    group.add_argument(
        "--ans2label-file", type=str, default=None, help="ans2label file"
    )
    group.add_argument(
        "--val-inference-type",
        type=str,
        default="allcand",
        help="inference type in validation (allcand or beamsearch), default to allcand",
    )
    return parser


def add_generator_args(parser):
    group = parser.add_argument_group("generator arguments", "configurations")
    group.add_argument("--beam", type=int, default=5, help="beam size")
    group.add_argument("--max-len-a", type=int, default=0, help="max-len-a")
    group.add_argument("--max-len-b", type=int, default=200, help="max-len-b")
    group.add_argument("--min-len", type=int, default=1, help="min-len")
    group.add_argument(
        "--no-repeat-ngram-size", type=int, default=0, help="no_repeat_ngram_size"
    )
    return parser


def get_args(add_custom_args_fn=add_custom_args):
    """Parse all the args."""
    parser = argparse.ArgumentParser(description="PyTorch Otter Model")
    parser = add_training_args(parser)
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    parser = add_distill_args(parser)
    parser = add_criterions_args(parser)
    parser = add_generator_args(parser)

    if add_custom_args_fn is not None:
        parser = add_custom_args_fn(parser)

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))

    if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
        # We are using (OpenMPI) mpirun for launching distributed data parallel processes
        local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
        # local_rank = args.rank % torch.cuda.device_count()
        print("local rank {}".format(local_rank))
        local_size = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE"))
        # local_size = torch.cuda.device_count()

        # Possibly running with Slurm
        num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES", "1"))
        nodeid = int(os.getenv("SLURM_NODEID", "0"))

        args.local_rank = local_rank
        args.rank = nodeid * local_size + local_rank
        args.world_size = num_nodes * local_size

    args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print(
            "using world size: {} and model-parallel size: {} ".format(
                args.world_size, args.model_parallel_size
            )
        )

    if args.micro_batch_size <= 0:
        args.micro_batch_size = args.batch_size
    assert args.batch_size % args.micro_batch_size == 0

    return args
