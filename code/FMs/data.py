import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def context_data_load():
    data_path = "../data/"

    train_path = os.path.join(data_path, "train_data.csv")
    test_path = os.path.join(data_path, "test_data.csv")
    sub_path = os.path.join(data_path, "sample_submission.csv")

    train = pd.read_csv(train_path, parse_dates=["Timestamp"])
    test = pd.read_csv(test_path, parse_dates=["Timestamp"])
    sub = pd.read_csv(sub_path)

    # test에서 -1아닌 부분 train으로 변경
    drop_index_test = test[test["answerCode"] == -1].index
    test2train = test.drop(drop_index_test)
    train = pd.concat([train, test2train])

    # test에서 -1아닌 부분 drop
    remain_index_test = test[test["answerCode"] != -1].index
    test = test.drop(remain_index_test)

    ids = train["userID"].unique()
    items = train["assessmentItemID"].unique()

    # RS 모델은 중복을 허용하지 않으므로 중복 레코드는 최종 레코드만을 보존
    train = train.drop_duplicates(subset=["userID", "assessmentItemID"], keep="last")
    train = train.reset_index(drop=True)
    idx, train, test = process_context_data(train, test)

    idx2user = {idx: id for idx, id in enumerate(ids)}
    idx2item = {idx: item for idx, item in enumerate(items)}

    user2idx = {id: idx for idx, id in idx2user.items()}
    item2idx = {item: idx for idx, item in idx2item.items()}

    train["userID"] = train["userID"].map(user2idx)
    train["assessmentItemID"] = train["assessmentItemID"].map(item2idx)

    test["userID"] = test["userID"].map(user2idx)
    test["assessmentItemID"] = test["assessmentItemID"].map(item2idx)

    field_dims = np.array(
        [
            len(user2idx),
            len(item2idx),
            len(idx["tag2idx"]),
            len(idx["timeDivide2idx"]),
            len(idx["grade2idx"]),
            len(idx["assessmentItemNumber2idx"]),
            len(idx["tagCount2idx"]),
        ],
        dtype=np.uint32,
    )
    data = {
        "train": train,
        "test": test.drop(["answerCode"], axis=1),
        "field_dims": field_dims,
        "sub": sub,
        "idx2user": idx2user,
        "idx2item": idx2item,
        "user2idx": user2idx,
        "item2idx": item2idx,
    }
    return data


def process_context_data(train, test):
    # 시간대를 4개로 나눔
    def get_time_divide(time):
        time = int(time)
        if time >= 0 and time < 6:
            return "dawn"
        elif time >= 6 and time < 12:
            return "morning"
        elif time >= 12 and time < 18:
            return "after_noon"
        else:
            return "evening"

    train["timeDivide"] = train["Timestamp"].map(lambda x: get_time_divide(x.hour))
    test["timeDivide"] = test["Timestamp"].map(lambda x: get_time_divide(x.hour))
    # train['timeDivide'] = train['timeDivide'].astype('category')
    timeDivide2idx = {v: k for k, v in enumerate(train["timeDivide"].unique())}
    train["timeDivide"] = train["timeDivide"].map(timeDivide2idx)
    test["timeDivide"] = test["timeDivide"].map(timeDivide2idx)

    # 시험지의 등급을 나눔
    train["grade"] = train["testId"].map(lambda x: int(x[2]))
    test["grade"] = test["testId"].map(lambda x: int(x[2]))
    # train['grade'] = train['grade'].astype('category')
    grade2idx = {v: k for k, v in enumerate(train["grade"].unique())}
    train["grade"] = train["grade"].map(grade2idx)
    test["grade"] = test["grade"].map(grade2idx)

    # 문제 번호 추출
    train["assessmentItemNumber"] = train["assessmentItemID"].map(lambda x: int(x[-3:]))
    test["assessmentItemNumber"] = test["assessmentItemID"].map(lambda x: int(x[-3:]))
    # train['assessmentItemNumber'] = train['assessmentItemNumber'].astype('category')
    assessmentItemNumber2idx = {
        v: k for k, v in enumerate(train["assessmentItemNumber"].unique())
    }
    train["assessmentItemNumber"] = train["assessmentItemNumber"].map(
        assessmentItemNumber2idx
    )
    test["assessmentItemNumber"] = test["assessmentItemNumber"].map(
        assessmentItemNumber2idx
    )

    # 저번에 봤던 태그의 문제인가?
    train["tagCount"] = train.groupby(["userID", "KnowledgeTag"]).cumcount() + 1
    test["tagCount"] = test.groupby(["userID", "KnowledgeTag"]).cumcount() + 1
    # train['tagCount'] = train['tagCount'].astype('category')
    tagCount2idx = {v: k for k, v in enumerate(train["tagCount"].unique())}
    train["tagCount"] = train["tagCount"].map(tagCount2idx)
    test["tagCount"] = test["tagCount"].map(tagCount2idx)

    # 태그 인덱싱
    tag2idx = {v: k for k, v in enumerate(train["KnowledgeTag"].unique())}
    train["KnowledgeTag"] = train["KnowledgeTag"].map(tag2idx)
    test["KnowledgeTag"] = test["KnowledgeTag"].map(tag2idx)

    # 안쓰는 열 삭제
    train = train.drop(columns=["Timestamp", "testId"])
    test = test.drop(columns=["Timestamp", "testId"])

    idx = {
        "timeDivide2idx": timeDivide2idx,
        "grade2idx": grade2idx,
        "assessmentItemNumber2idx": assessmentItemNumber2idx,
        "tagCount2idx": tagCount2idx,
        "tag2idx": tag2idx,
    }

    return idx, train, test


def last_solved_data_split(data):
    train = data["train"]
    validation_set = train.groupby("userID").tail(3)

    # validation_set에서 중복된게 있을 경우 삭제
    validation_set = validation_set.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last"
    )
    validation_set = validation_set.reset_index(drop=True)

    train_set = train.drop(validation_set.index)
    X_train = train_set.drop(columns=["answerCode"])
    X_valid = validation_set.drop(columns=["answerCode"])
    y_train = train_set["answerCode"]
    y_valid = validation_set["answerCode"]

    data["X_train"], data["X_valid"], data["y_train"], data["y_valid"] = (
        X_train,
        X_valid,
        y_train,
        y_valid,
    )
    return data


def context_data_loader(data):
    train_dataset = TensorDataset(
        torch.LongTensor(data["X_train"].values),
        torch.LongTensor(data["y_train"].values),
    )
    valid_dataset = TensorDataset(
        torch.LongTensor(data["X_valid"].values),
        torch.LongTensor(data["y_valid"].values),
    )
    test_dataset = TensorDataset(torch.LongTensor(data["test"].values))

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    data["train_dataloader"], data["valid_dataloader"], data["test_dataloader"] = (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
    )

    return data
