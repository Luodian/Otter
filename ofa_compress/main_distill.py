import os
import numpy as np
import random
import pathlib
import torch
import transformers
import logging
import pipeline

from functools import partial
from train import OFADistiller
from textbrewer import TrainingConfig
from arguments import get_args
from datetime import datetime
from evaluation import ddp_evaluate
from pipeline import initialize_distributed
from init_task import build_task



def args_check(args, logger):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.info("Output directory () already exists and is not empty.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(args.gradient_accumulation_steps))

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1

    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu


def main():
    import PIL
    print("PIL version", PIL.__version__)
    print("CUDA device name", torch.cuda.get_device_name(0))
    # ---------- SETTINGS ----------------
    torch.backends.cudnn.enabled = True
    args = get_args()

    cur_time = datetime.strftime(datetime.now(), '%m%d_%H%M')
    if args.postfix is not None:
        args.output_dir = os.path.join(args.output_dir, args.postfix)
    folder_name = args.task + "/" + cur_time
    args.output_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(args.output_dir, exist_ok=True)
    log_output_dir = args.output_dir
    os.makedirs(log_output_dir, exist_ok=True)
    if args.rank == 0:
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    os.path.join(log_output_dir, f'log_{args.rank}.txt'))
            ],
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        logger = logging.getLogger("Main")
        print(logger)
    else:
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            handlers=[logging.StreamHandler()],
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        logger = logging.getLogger("Main")
        print(logger)


    logger.info("---------- SETTINGS ----------------")
    initialize_distributed(args)


    logger.warning(
        f'args.world_size = {args.world_size}, args.rank ={args.rank}, args.local_rank = {args.local_rank}'
    )
    device, n_gpu = args_check(args, logger)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Args : {args.__dict__}")

    logger.info("---------- RANDOM SEEDs ----------------")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    build_task(args)

    # ---------- DATA LOADER ----------------
    logger.info("---------- DATA LOADER ----------------")

    train_loader, val_loader = pipeline.get_data_loader(args, logger)
    logger.info(f'train loader: {train_loader}, length: {len(train_loader)}')
    if len(args.train_dataset) > 1:
        args.num_train_steps = sum([len(
            train_loader[i]) for i in range(len(args.train_dataset))]) // args.gradient_accumulation_steps * args.num_epochs
    else:
        args.num_train_steps = len(
            train_loader) // args.gradient_accumulation_steps * args.num_epochs

    # ---------- MODELs ----------------
    logger.info("----------- MODELS ------------")
    model_T = pipeline.get_teacher_model(args)
    model_S = pipeline.get_student_model(args)
    args.model_T_config = model_T.config.to_dict()
    args.model_S_config = model_S.config.to_dict()
    logger.info(f'Config of teacher model: {model_T.config.to_dict()}')
    logger.info(' Number of teacher model parameters on rank {}: {}'.format(
        torch.distributed.get_rank(),
        sum([p.nelement() for p in model_T.parameters()])))
    logger.info(f'Config of student model: {model_S.config.to_dict()}')
    logger.info(' Number of student model parameters on rank {}: {}'.format(
        torch.distributed.get_rank(),
        sum([p.nelement() for p in model_S.parameters()])))
    model_T.to(device)
    model_S.to(device)
    adaptor_T = pipeline.get_adaptors(args)
    adaptor_S = pipeline.get_adaptors(args)

    # ---------- TRAIN ----------------
    if args.do_train:
        logger.info("---------- TRAIN ----------------")
        logger.info(model_S.parameters())
        optimizer = transformers.AdamW(
            model_S.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            correct_bias=False)

        scheduler_class, scheduler_args = pipeline.get_schedule(args)
        if args.ckpt_frequency > -1:
            train_config = TrainingConfig(
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                ckpt_frequency=args.ckpt_frequency,
                log_dir=args.output_dir,
                output_dir=args.output_dir,
                device=args.device,
                fp16=args.fp16,
                local_rank=args.local_rank)
        else:
            train_config = TrainingConfig(
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                ckpt_epoch_frequency=args.ckpt_epoch_frequency,
                log_dir=args.output_dir,
                output_dir=args.output_dir,
                device=args.device,
                fp16=args.fp16,
                local_rank=args.local_rank)

        logger.info(f"Train_config:\n {train_config}")

        # Distill Config
        distill_config = pipeline.get_distill_config(args)
        logger.info(f"Distill config:\n {distill_config}")

        distiller = OFADistiller(train_config, distill_config, model_T,
                                     model_S, adaptor_T, adaptor_S)
        # ---------- EVAL ----------------
        args.evaluate_idx = 0
        callback_func = partial(ddp_evaluate,
                                eval_dataloader=val_loader,
                                args=args,
                                logger=logger)


        def batch_postprocessor(batch):
            return batch

        with distiller:
            distiller.train(optimizer,
                            scheduler_class=scheduler_class,
                            scheduler_args=scheduler_args,
                            max_grad_norm=1.0,
                            dataloader=train_loader,
                            num_epochs=args.num_epochs,
                            callback=callback_func,
                            batch_postprocessor=batch_postprocessor)


if __name__ == '__main__':
    main()
