import os

import numpy as np
import pandas as pd

from boosting.args import parse_args
from boosting.data_loader import Dataset, Preprocess
from boosting.utils import set_seeds
from boosting.boosting import boosting_model

import warnings

warnings.filterwarnings("ignore")


def main(args):
    ######################## DATA LOAD
    print("Load Data")
    train = pd.read_csv(args.data_dir + args.file_name, parse_dates=["Timestamp"])
    test = pd.read_csv(args.data_dir + args.test_file_name, parse_dates=["Timestamp"])

    data = Dataset(train, test)
    data = data.split_data()
    print("Succesfully Split Data")

    ######################## SELECT FEATURE

    FEATURE = ["userID", "assessmentItemID", "Timestamp", "KnowledgeTag"]

    ######################## DATA PREPROCESSING

    print("Start Preprocessing Data")
    process = Preprocess(args, data, FEATURE)
    data = process.preprocess()
    print("Succesfully Preprocess Data")

    ######################## MODEL INIT

    model = boosting_model(args, FEATURE)

    #     ######################## TRAIN
    model.training(data, args)
    model.inference(data)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
