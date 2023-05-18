import os

import numpy as np
import pandas as pd

from boosting.args import parse_args
from boosting.dataloader import Dataset, Preprocess
from boosting.utils import get_logger, set_seeds, logging_conf
from boosting.model import boosting_model


logger = get_logger(logging_conf)


def main(args):
    ######################## DATA LOAD
    print("Load Data")
    train = pd.read_csv(args.data_dir + args.file_name)
    test = pd.read_csv(args.data_dir + args.test_file_name)

    data = Dataset(train, test)
    data = data.split_data()
    print("Succesfully Split Data")

    ######################## SELECT FEATURE

    FEATURE = []

    ######################## DATA PREPROCESSING

    data = Preprocess(data, FEATURE)

    ######################## MODEL INIT

    model = boosting_model(args, FEATURE)

    ######################## TRAIN

    model.training(data)

    model.inference(data)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
