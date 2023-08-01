import os
import random
import numpy as np
import json
import torch
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import argparse


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")
    parser.add_argument(
        "--data_dir",
        default="../data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )
    parser.add_argument(
        "--train_file_name", default="train_data.csv", type=str, help="train file name"
    )
    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="best_model.pt", type=str, help="model file name"
    )
    parser.add_argument(
        "--save_dir", default="saved_models/", type=str, help="output directory"
    )
    parser.add_argument(
        "--submit_dir", default="submit/", type=str, help="submit directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    # Model(LSTM, Transformer..)
    parser.add_argument("--n_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--mlp_dim", default=64, type=int, help="number of heads")
    parser.add_argument("--use_each_position", default=False, type=bool, help="Positional Embedding in Transformer")
    parser.add_argument(
        "--max_seq_len", default=20, type=int, help="max sequence length"
    )

    # Model(GCN)
    parser.add_argument("--alpha", default=None, type=float, help="alpha")

    # Model(공통)
    parser.add_argument("--activation", default="sigmoid", type=str, help="model type")
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size"
    )
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers")
    parser.add_argument("--drop_out", default=0.2, type=float, help="drop out rate")

    # train
    parser.add_argument(
        "--loss", default="BCE_sigmoid_loss", type=str, help="loss function"
    )
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )
    parser.add_argument("--n_epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument("--clip_grad", default=10, type=int, help="clip grad")
    parser.add_argument("--patience", default=5, type=int, help="for early stopping")
    parser.add_argument("--factor", default=0.5, type=float, help="optimizer factor")
    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    # Data Loader
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--shuffle", default=True, type=bool, help="shuffle")
    parser.add_argument(
        "--split_ratio", default=0.2, type=float, help="train/valid ratio"
    )
    parser.add_argument("--num_workers", default=2, type=int, help="number of workers")

    # 중요
    parser.add_argument("--dataset", default="DKTDataset", type=str, help="dataset")
    parser.add_argument("--model", default="LSTM", type=str, help="model type")
    parser.add_argument(
        "--data_loader", default="LSTMDataLoader", type=str, help="data loader"
    )
    parser.add_argument(
        "--trainer", default="LSTMTrainer", type=str, help="data trainer"
    )

    args = parser.parse_args()

    return args


def set_seeds(seed: int = 42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(logger_conf: dict):
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}
