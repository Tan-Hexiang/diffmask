#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from diffmask.models.question_answering_nq_diffmask import (
    FidQuestionAnsweringNQDiffMask
)
import logging
logging.basicConfig(level=logging.DEBUG,  format='%(name)s - %(levelname)s - %(message)s')

# from diffmask.utils.callbacks import CallbackNQDiffMask


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--passage_len", type=int, default=200)
    parser.add_argument("--target_len", type=int, default=20)
    parser.add_argument("--n_context",type=int,default=100)
    parser.add_argument("--gpus", type=str, default="1")
    # parser.add_argument("--model", type=str, default="bert-large-uncased-whole-word-masking-finetuned-squad")
    parser.add_argument(
        "--train_filename",
        type=str,
        default="/data/tanhexiang/tevatron/tevatron/data_nq/result100/fid.nq.small.jsonl",
    )
    parser.add_argument(
        "--val_filename",
        type=str,
        default="/data/tanhexiang/tevatron/tevatron/data_nq/result100/fid.nq.small.jsonl",
    )
    parser.add_argument(
        "--passages_source_path",
        type=str,
        default="/data/tanhexiang/CF_QA/data/wikipedia_split/psgs_w100.tsv"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/tanhexiang/CF_QA/models/reader/nq_reader_base",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--gate_bias", action="store_true")
    parser.add_argument("--learning_rate_alpha", type=float, default=3e-1)
    parser.add_argument("--learning_rate_placeholder", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1)
    parser.add_argument("--eps_valid", type=float, default=3)
    parser.add_argument("--acc_valid", type=float, default=0.0)
    # 默认开启！！！！
    parser.add_argument("--placeholder", action="store_false")
    # parser.add_argument("--stop_train", action="store_true")
    parser.add_argument(
        "--gate",
        type=str,
        default="input",
        choices=["input", "hidden", "per_sample-reinforce", "per_sample-diffmask"],
    )
    parser.add_argument("--layer_pred", type=int, default=-1)
    
    hparams= parser.parse_args()
    
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpu

    model = FidQuestionAnsweringNQDiffMask(hparams)
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.getcwd(),
        version=None,
        name='lightning_logs'
    )
    trainer = pl.Trainer(
        gpus=int(hparams.gpu != ""),
        progress_bar_refresh_rate=1,
        max_epochs=hparams.epochs,
        # callbacks=[CallbackNQDiffMask()],
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                "outputs",
                "nq-fid-{}-layer_pred={}".format(hparams.gate, hparams.layer_pred),
                "{epoch}-{val_loss:.2f}-{loss_g:.2f}-{val_l0:.2f}",
            ),
            verbose=True,
            save_top_k=50
        ),
        logger=tb_logger
    )

    trainer.fit(model)
