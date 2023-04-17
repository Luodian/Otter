import os
import torch
import adaptors
import transformers
from torch.utils.data import SequentialSampler
from ofa.configuration_ofa import OFAConfig
from ofa.modeling_ofa import OFAModel

from model_paths import teacher_model_paths,student_model_paths
from matches import generate_matches, generate_matches_ofa
from architecture_configs import architecture_configs_dict
from train import OFADistillationConfig
from data_utils import OFADataset, CaptionDataset, UnifyDataset, RefcocoDataset, SnliVeDataset,VqaGenDataset
from torch.utils.data.distributed import DistributedSampler
import logging
logger = logging.getLogger("Pipeline")

TOKENIZER_PATH = "./tokenizer"

GLUE_TASKS = [
    "mrpc", "sst2", "qqp", "qnli", "rte", "cola", "stsb", "mnli_mismatched",
    "mnli_matched"
]
OFA_TASKS = ["caption_stage1", "caption_stage2"]

def get_data_loader(args, logger):
    logger.info("Get data loader")

    # Data parallel arguments.
    num_workers = torch.distributed.get_world_size()

    if args.task in ["caption_stage1", "caption_stage2"]:
        dataset_class = CaptionDataset
    elif args.task == "pretrain":
        dataset_class = UnifyDataset
    elif args.task in ["refcoco", "refcocog", "refcocoplus"]:
        dataset_class = RefcocoDataset
    elif args.task == "snli_ve":
        dataset_class = SnliVeDataset
    elif args.task == "vqa_gen":
        dataset_class = VqaGenDataset
    else:
        dataset_class = OFADataset
    # Build the dataset.
    dataset = [dataset_class(args,
                            args.train_dataset[i],
                            ) for i in range(len(args.train_dataset))]
    logger.info(f"Dataset:{dataset}")
    if args.num_epochs > 0:
        total_samples = sum([len(dataset[i]) for i in range(len(args.train_dataset))])
        samples_per_iter = args.batch_size * num_workers
        ori_train_iters = args.train_iters
        args.train_iters = (args.num_epochs * total_samples +
                            samples_per_iter - 1) // samples_per_iter
        logger.info(
            f'modified args.train_iters from {ori_train_iters} to {args.train_iters}, due to num epochs = {args.num_epochs}'
        )
        logger.info(
            '> num_epochs=%d samples=%d workers=%d batch=%d samples_per_iter=%d train_iters=%d'
            % (args.num_epochs, total_samples, num_workers, args.batch_size,
               samples_per_iter, args.train_iters))
    # Use a simple sampler with distributed batch sampler.
    sampler = [DistributedSampler(dataset[i]) for i in range(len(args.train_dataset))]

    samples_per_ga_step = args.micro_batch_size * num_workers
    logger.info(f"samples_per_ga_step: {samples_per_ga_step}")

    # Torch data loader.
    train_data_loader = [torch.utils.data.DataLoader(dataset[i],
                                                    sampler=sampler[i],
                                                    collate_fn=dataset[i].collate,
                                                    batch_size=args.batch_size,
                                                    drop_last=True,
                                                    num_workers=0,
                                                    pin_memory=True) for i in range(len(args.train_dataset))]



    eval_dataset = dataset_class(args,
                                 args.test_dataset,
                                 is_test=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        collate_fn=eval_dataset.collate,
        num_workers=0,
        pin_memory=True,
        batch_size=args.batch_size)


    return train_data_loader, eval_data_loader


def get_teacher_model(args):
    if args.load_teacher_model:
        if args.task in teacher_model_paths:
            teacher_model_path = teacher_model_paths[args.task]
        else:
            teacher_model_path = args.load_teacher_model
        print(f"Load Teacher Model from {teacher_model_path}")
        if args.intermediate_matches:
            output_dict = {
                "output_attentions": True,
                "output_hidden_states": True
            }
        else:
            output_dict = {}
        model_T = OFAModel.from_pretrained(teacher_model_path, **output_dict)
    else:
        model_T = None
    return model_T


def get_student_model(args):
    load_model = args.load_student_model
    init_method = args.init_method
    print(init_method)
    if args.intermediate_matches:
        custom_dict = {
            "output_attentions": True,
            "output_hidden_states": True
        }
    else:
        custom_dict = {}
    if args.generator_version == 'fairseq':
        custom_dict.update({"use_cache": True})
    else:
        custom_dict.update({"use_cache": False})

    if init_method in ['load_pretrain', 'load_distilled']:
        print(f"Load student model from {load_model}.")
        if load_model in student_model_paths[init_method]:
            model_path = student_model_paths[init_method][load_model]
            logger.info(f"Load student model from {model_path}.")
        else:
            raise NotImplementedError(f"Model {load_model} not found")
        model_S = OFAModel.from_pretrained(model_path, **custom_dict)
    elif init_method in ['random']:
        print("Create a student model from scratch.")
        logger.info("Create a student model from scratch.")
        configs = architecture_configs_dict[args.student_model_config]
        config = configs['config']
        config.update(custom_dict)
        model_config = OFAConfig(**config)
        model_S = OFAModel(model_config)
    print('architecture of model_S', model_S)
    return model_S

def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    print("device id: {}".format(device))
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(backend=args.distributed_backend,
                                         world_size=args.world_size,
                                         rank=args.rank,
                                         init_method=init_method)
    print('args.world_size =', args.world_size, ', args.rank =', args.rank,
          ', args.local_rank =', args.local_rank)
    assert args.rank == torch.distributed.get_rank()

def get_schedule(args):

    if args.schedule == 'const':
        scheduler_class = transformers.get_constant_schedule_with_warmup
        scheduler_args = {
            'num_warmup_steps':
            int(args.warmup_proportion * args.num_train_steps)
        }
    elif args.schedule == 'linear':
        scheduler_class = transformers.get_linear_schedule_with_warmup
        scheduler_args = {
            'num_warmup_steps':
            int(args.warmup_proportion * args.num_train_steps),
            'num_training_steps': args.num_train_steps
        }
    elif args.schedule == 'cosine':
        scheduler_class = transformers.get_cosine_schedule_with_warmup
        scheduler_args = {
            'num_warmup_steps':
            int(args.warmup_proportion * args.num_train_steps),
            'num_training_steps': args.num_train_steps
        }
    elif args.schedule == 'polynomial_decay':
        scheduler_class = transformers.get_polynomial_decay_schedule_with_warmup
        scheduler_args = {
            'num_warmup_steps':
            int(args.warmup_proportion * args.num_train_steps),
            'num_training_steps': args.num_train_steps,
            'lr_end': args.lr_end
        }
    else:
        raise NotImplementedError

    return scheduler_class, scheduler_args

def get_adaptors(args):
    if args.task in ['caption_stage1', 'refcoco', 'refcocog', 'refcocoplus', 'snli_ve', 'vqa_gen']:
        return adaptors.FinetuneAdaptor(args)
    elif args.task == 'pretrain':
        return adaptors.PretrainAdaptor(args)
    else:
        raise NotImplementedError


def get_distill_config(args):
    if args.temperature_scheduler == 'flsw':
        args.temperature_scheduler = [
            'flsw',
            float(args.temperature_beta),
            float(args.temperature_gamma)
        ]
    elif args.temperature_scheduler == 'cwsm':
        args.temperature_scheduler = ['cwsm', float(args.temperature_beta)]

    # Intermediate matches config
    match_name = args.intermediate_matches
    if match_name:
        args.intermediate_matches = generate_matches_ofa(args)
    distill_config = OFADistillationConfig(
        text_preprocessor=args.tokenizer,
        temperature=args.temperature,
        temperature_scheduler=args.temperature_scheduler,
        hard_label_weight=args.hard_label_weight,
        hard_label_weight_scheduler=args.hard_label_weight_scheduler,
        kd_loss_type=args.kd_loss_type,
        kd_loss_weight=args.kd_loss_weight,
        kd_loss_weight_scheduler=args.kd_loss_weight_scheduler,
        probability_shift=args.probability_shift,
        intermediate_matches=args.intermediate_matches,
        is_caching_logits=args.is_caching_logits,
        constraint_range=args.constraint_range)  # constraint_range
    return distill_config
