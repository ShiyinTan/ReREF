from numpy import NaN
import torch
import os
import argparse
from transformers import Adafactor
from tqdm import tqdm

import pandas as pd
import pdb
# from primer_summarizer_module import RankModel
from rank_only_model import RankModel
# from datasets import load_dataset, load_metric
from datasets import load_dataset
import evaluate
import json
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    LEDTokenizer,
    LEDForConditionalGeneration,
)
from dataloader_new import ( # from dataloader_new
    get_dataloader_summ,
    get_dataloader_summiter,
)
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
# from pytorch_lightning.plugins import DDPPlugin
from pathlib import Path
import torch.distributed as dist
from itertools import chain

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def train(args):
    args.compute_accuracy = True
    model = RankModel(args)
    tokenizer = AutoTokenizer.from_pretrained(args.primer_path)
    # initialize checkpoint
    if args.ckpt_path is None:
        args.ckpt_path = args.model_path + "checkpoints/"

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="{step}-{vloss:.2f}-{precision:.4f}",
        save_top_k=args.saveTopK,
        monitor="precision",
        mode="max",
        save_last=False,
        save_on_train_epoch_end=False,
    )

    tqdm_progbar_callback = TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate * args.acc_batch)
    
    # initialize logger
    logger = TensorBoardLogger(args.model_path + "tb_logs", name="my_model")

    # initialize trainer
    pl.seed_everything(args.rand_seed, workers=True)
    trainer = pl.Trainer(
        devices=args.devices if args.devices!=0 else 'auto',
        max_steps=args.total_steps,
        accumulate_grad_batches=args.acc_batch,
        check_val_every_n_epoch=1 if args.num_train_data > 100 else 1,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, tqdm_progbar_callback],
        enable_checkpointing=True,
        enable_progress_bar=True,
        precision='32', # 32
        accelerator=args.accelerator,
        strategy = args.strategy, 
    )

    rank_hf_datasets = load_dataset(f"HF-Data-for-Retriever/{args.dataset_name}")
    train_dataloader = get_dataloader_summ(
        args, rank_hf_datasets, tokenizer, "train", args.num_workers, True
    )
    valid_dataloader = get_dataloader_summ(
        args, rank_hf_datasets, tokenizer, "validation", args.num_workers, False
    )
    test_dataloader = get_dataloader_summ(
        args, rank_hf_datasets, tokenizer, "test", args.num_workers, False
    )

    trainer.fit(model, train_dataloader, valid_dataloader)
    if args.test_imediate:
        args.resume_ckpt = checkpoint_callback.best_model_path
        print(args.resume_ckpt)
        if args.test_batch_size != -1:
            args.batch_size = args.test_batch_size
        args.mode = "test"
        test(args)


def test(args):
    args.compute_accuracy = True
    tokenizer = AutoTokenizer.from_pretrained(args.primer_path)
    tqdm_progbar_callback = TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)
    # initialize trainer
    trainer = pl.Trainer(
        devices=args.devices if args.devices!=0 else 'auto',

        max_steps = -1, 
        log_every_n_steps=5,
        enable_checkpointing=False, # True
        callbacks=[tqdm_progbar_callback],
        enable_progress_bar=True,
        precision='32', # 32
        accelerator=args.accelerator,
        strategy=args.strategy, 
    )

    if args.resume_ckpt is not None:
        model = RankModel.load_from_checkpoint(args.resume_ckpt, args=args)
        print("load from checkpoint")
    else:
        model = RankModel(args)


    # load dataset
    rank_hf_datasets = load_dataset(f"HF-Data-for-Retriever/{args.dataset_name}")
    test_dataloader = get_dataloader_summ(
        args, rank_hf_datasets, tokenizer, "test", args.num_workers, False
    )
    

    # test
    trainer.test(model, test_dataloader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ########################
    # Gneral
    parser.add_argument("--devices", default=0, type=int, help="number of gpus to use")
    parser.add_argument(
        "--accelerator", default='gpu', type=str, help="Type of accelerator"
    ) # gpu
    parser.add_argument(
        "--strategy", default='auto', type=str, help="Whether to use ddp, ddp_spawn strategy"
    ) # gpu, ddp_find_unused_parameters_true
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument(
        "--model_name", default="primer",
    )
    parser.add_argument(
        "--primer_path", type=str, default="allenai/PRIMERA", # allenai/PRIMERA
    )
    parser.add_argument("--join_method", type=str, default="concat_start_wdoc_global") # concat_start_wdoc_global
    parser.add_argument(
        "--debug_mode", action="store_true", help="set true if to debug"
    )
    parser.add_argument(
        "--compute_accuracy",
        action="store_true",
        help="whether to compute rank accuracy in validation steps",
    )
    parser.add_argument(
        "--saveRouge",
        action="store_true",
        help="whether to compute rouge in validation steps",
    )

    parser.add_argument("--progress_bar_refresh_rate", default=10, type=int)
    parser.add_argument("--model_path", type=str, default="./run_saves/save_path/") # "./pegasus/"
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--saveTopK", default=3, type=int)
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        help="Path of a checkpoint to resume from",
        default=None,
    )

    parser.add_argument("--data_path", type=str, default="../dataset/")
    parser.add_argument("--dataset_name", type=str, default="multi_news") # arxiv
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers to use for dataloader",
    )

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_length_input", default=512, type=int)
    parser.add_argument("--max_length_tgt", default=1024, type=int)
    parser.add_argument("--chunk_size", default=1024, type=int)
    parser.add_argument("--min_length_tgt", default=0, type=int)
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument(
        "--adafactor", action="store_true", help="Use adafactor optimizer"
    )
    parser.add_argument(
        "--grad_ckpt",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=0,
        help="seed for random sampling, useful for few shot learning",
    )

    ########################
    # For training
    parser.add_argument(
        "--pretrained_model_path", type=str, default="./pretrained_models/",
    )
    parser.add_argument(
        "--loss_type", type=str, default="bpr", help="Type of loss to use" # rll
    )
    parser.add_argument(
        "--model_encoder", type=str, default="encoder_decoder", help="Type of loss to use" # encoder_only, encoder_decoder
    )
    parser.add_argument(
        "--limit_valid_batches", type=int, default=None,
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Maximum learning rate") # 3e-5
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Number of warmup steps" # 5000
    )
    parser.add_argument(
        "--accum_data_per_step", type=int, default=16, help="Number of data per step" # 16
    )
    parser.add_argument(
        "--total_steps", type=int, default=1000000, help="Number of steps to train" # 50000
    )
    parser.add_argument(
        "--num_train_data",
        type=int,
        default=-1,
        help="Number of training data, -1 for full dataset and any positive number indicates how many data to use",
    )

    parser.add_argument(
        "--fix_lr", action="store_true", help="use fix learning rate",
    )
    parser.add_argument(
        "--test_imediate", action="store_true", help="test on the best checkpoint",
    )
    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="whether this is a run for few shot learning",
    )
    ########################
    # For testing
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="Number of batches to test in the test mode.",
    )
    parser.add_argument("--beam_size", type=int, default=1, help="size of beam search")
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1,
        help="length penalty of generated text",
    )
    parser.add_argument(
        "--mask_num",
        type=int,
        default=0,
        help="Number of masks in the input of summarization data",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=-1,
        help="batch size for test, used in few shot evaluation.",
    )
    parser.add_argument(
        "--applyTriblck",
        action="store_true",
        help="whether apply trigram block in the evaluation phase",
    )

    args = parser.parse_args()  # Get pad token id
    ####################
    if args.accum_data_per_step == -1:
        args.acc_batch = 1
    else:
        args.acc_batch = args.accum_data_per_step // args.batch_size
    args.data_path = os.path.join(args.data_path, args.dataset_name)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    

    if args.dataset_name == "multi_news":
        args.primer_path = "allenai/PRIMERA-multinews"
    elif args.dataset_name == "multi_x_science_sum":
        args.primer_path = "allenai/PRIMERA-multixscience"
    elif args.dataset_name == "wcep":
        args.primer_path = "allenai/PRIMERA-wcep"


    print(args)
    with open(
        os.path.join(
            args.model_path, "args_%s_%s.json" % (args.mode, args.dataset_name)
        ),
        "w",
    ) as f:
        json.dump(args.__dict__, f, indent=2)

    if args.mode == "train":
        train(args)
    else:
        test(args)
