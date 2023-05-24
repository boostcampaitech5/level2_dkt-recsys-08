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

    FEATURE = [
        "userID",
        "assessmentItemID",
        "KnowledgeTag",
        "elapsed",
        "category_high",
        "problem_num",
        "cum_answerRate_per_user",
        "acc_elapsed_per_user",
        "problem_correct_per_user",
        "problem_solved_per_user",
        "correct_answer_per_cat",
        "acc_count_per_cat",
        "acc_answerRate_per_cat",
        "timeDelta_userAverage",
        "timestep_1",
        "timestep_2",
        "timestep_3",
        "timestep_4",
        "timestep_5",
        "hour",
        "weekofyear",
        "problem_correct_per_woy",
        "problem_solved_per_woy",
        "cum_answerRate_per_woy",
    ]
    FEATURE_USER = [
        "answerRate_per_user",
        "answer_cnt_per_user",
        "elapsed_time_median_per_user",
        "assessment_solved_per_user",
    ]
    FEATURE_ITEM = [
        "answerRate_per_item",
        "answer_cnt_per_item",
        "elapsed_time_median_per_item",
        "wrong_users_median_elapsed",
        "correct_users_median_elapsed",
    ]
    FEATURE_TAG = ["tag_exposed", "answerRate_per_tag"]
    FEATURE_TEST = ["elapsed_median_per_test", "answerRate_per_test"]
    FEATURE_CAT = ["elapsed_median_per_cat", "answerRate_per_cat"]
    FEATURE_PROBLEM_NUM = [
        "elapsed_median_per_problem_num",
        "answerRate_per_problem_num",
    ]

    # FEATURE_ELO = ["elo_assessment", "elo_test", "elo_tag"]

    FEATURE += FEATURE_USER
    FEATURE += FEATURE_ITEM
    FEATURE += FEATURE_TAG
    FEATURE += FEATURE_TEST
    FEATURE += FEATURE_CAT
    FEATURE += FEATURE_PROBLEM_NUM
    # FEATURE += FEATURE_ELO

    ######################## DATA PREPROCESSING

    print("Start Preprocessing Data")
    process = Preprocess(args, data)
    data = process.preprocess()
    print("Succesfully Preprocess Data")

    ######################## MODEL INIT
    print("number of selected features:", len(FEATURE))
    model = boosting_model(args, FEATURE)

    ######################## TRAIN

    model.training(data, args)

    ######################## INFERENCE

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.inference(data)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
