from base import BaseDataLoader
import os
import time
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from base import BaseDataLoader
from sklearn.preprocessing import LabelEncoder
from utils import get_logger, logging_conf
from sklearn.model_selection import train_test_split

logger = get_logger(logging_conf)


class FMDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        self.data_dir = data_dir
        self.dataset = self.__load_data_from_file(data_dir)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )

    def __preprocessing(self, train_data, test_data):
        # 중복 레코드 제거
        # RS 모델에서는 시간에 따른 변화를 고려하지 않기 때문에 최종 성적만을 바탕으로 평가한다.
        train_data.drop_duplicates(
            subset=["userID", "assessmentItemID"], keep="last", inplace=True
        )
        test_data.drop_duplicates(
            subset=["userID", "assessmentItemID"], keep="last", inplace=True
        )

        # 불필요한 column 제거
        train_data.drop(
            ["Timestamp", "testId", "KnowledgeTag"],
            axis=1,
            inplace=True,
            errors="ignore",
        )

        # 평가 항목 제거
        # test 파일의 유저별로 마지막에 -1이 있는데 이를 학습에 쓸 수 없으니 제거한다.
        test_data = test_data[test_data.answerCode >= 0].copy()

        # 평가 항목 신규 생성
        # 남은 테스트 항목 중, 각 사용자별 최종 레코드를 새로운 평가 항목으로 정한다.
        eval_data = test_data.copy()
        eval_data.drop_duplicates(subset=["userID"], keep="last", inplace=True)

        # 평가 항목을 테스트 항목에서 제거한다.
        # 약간 마지막 값을 validation set으로 구성하는 느낌인것 같음
        test_data.drop(index=eval_data.index, inplace=True, errors="ignore")

        # 사용자 - 문제 항목 관계를 pivot 테이블로 변경
        # 각 사용자 별로 해당 문제를 맞췄는지 여부를 matrix 형태로 변경
        # 해당 문제를 푼 적이 없는 경우 0으로 설정
        matrix_train = train_data.pivot_table(
            "answerCode", index="userID", columns="assessmentItemID"
        )
        matrix_train.fillna(0.5, inplace=True)

        # 사용자 - 문제 항목의 pivot table을 normalize된 matrix로 변경
        X = matrix_train.values
        a_mean = np.mean(X, axis=1)
        Xm = X - a_mean.reshape(-1, 1)

        return Xm

    def __load_data_from_file(self, data_dir):
        train_path = os.path.join(data_dir, "train_data.csv")
        test_path = os.path.join(data_dir, "test_data.csv")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        dataset = self.__preprocessing(train_data, test_data)
        return dataset


class DKTDataLoader(BaseDataLoader):
    def __init__(self, args, training=True):
        super().__init__()
        self.args = args
        self.data_dir = self.args.data_dir
        self.data = self.load_data_from_file(self.data_dir, training)
        self.data = self.feature_engineering(self.data)
        self.data = self.preprocessing(self.data, training)

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_tests = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tags = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )

    def split_data(self, args):
        """
        split data into two parts with a given ratio.
        """
        return train_test_split(
            self.data, test_size=args.split_ratio, random_state=args.seed, shuffle=True
        )

    def save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s: str):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                )
            )
        )

        return group.values

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed
        return df

    def load_data_from_file(self, data_dir, training):
        if training:
            data_dir = os.path.join(data_dir, self.args.train_file_name)
        else:
            data_dir = os.path.join(data_dir, self.args.test_file_name)
        df = pd.read_csv(data_dir)
        return df

    def get_loader(self, args, dataset, data, shuffle):
        dataset = dataset(data, args)
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=args.num_workers,
            shuffle=shuffle,
            batch_size=args.batch_size,
        )
        return loader


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        self.max_seq_len = args.max_seq_len

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]

        # Load from data
        test, question, tag, correct = row[0], row[1], row[2], row[3]
        data = {
            "test": torch.tensor(test + 1, dtype=torch.int),
            "question": torch.tensor(question + 1, dtype=torch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
        }

        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len :]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len - seq_len :] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask

        # Generate interaction
        interaction = data["correct"] + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}
        return data

    def __len__(self) -> int:
        return len(self.data)


class DKTDataset_split(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        self.max_seq_len = args.max_seq_len

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]

        # Load from data
        test, question, tag, correct = row[0], row[1], row[2], row[3]
        data = {
            "test": torch.tensor(test + 1, dtype=torch.int),
            "question": torch.tensor(question + 1, dtype=torch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
        }

        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len:]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len - seq_len:] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask

        # Generate interaction
        interaction = data["correct"] + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}
        return data

    def __len__(self) -> int:
        return len(self.data)

class GCNDataLoader:
    def __init__(self, args, training=True):
        data = self.load_data(args, data_dir=args.data_dir)
        train_data, test_data = self.separate_data(data=data)
        args.id2index: dict = self.indexing_data(data=data)
        args.train_data = self.process_data(
            data=train_data, id2index=args.id2index, device=args.device
        )
        args.test_data = self.process_data(
            data=test_data, id2index=args.id2index, device=args.device
        )

        self.print_data_stat(train_data, "Train")
        self.print_data_stat(test_data, "Test")

        args.n_node = len(args.id2index)

    def load_data(self, args, data_dir: str) -> pd.DataFrame:
        path1 = os.path.join(data_dir, args.train_file_name)
        path2 = os.path.join(data_dir, args.tetst_file_name)
        data1 = pd.read_csv(path1)
        data2 = pd.read_csv(path2)

        data = pd.concat([data1, data2])
        data.drop_duplicates(
            subset=["userID", "assessmentItemID"], keep="last", inplace=True
        )
        return data

    def separate_data(self, data: pd.DataFrame):
        train_data = data[data.answerCode >= 0]
        test_data = data[data.answerCode < 0]
        return train_data, test_data

    def indexing_data(self, data: pd.DataFrame) -> dict:
        userid, itemid = (
            sorted(list(set(data.userID))),
            sorted(list(set(data.assessmentItemID))),
        )
        n_user, n_item = len(userid), len(itemid)

        userid2index = {v: i for i, v in enumerate(userid)}
        itemid2index = {v: i + n_user for i, v in enumerate(itemid)}
        id2index = dict(userid2index, **itemid2index)
        return id2index

    def process_data(self, data: pd.DataFrame, id2index: dict, device: str) -> dict:
        edge, label = [], []
        for user, item, acode in zip(
            data.userID, data.assessmentItemID, data.answerCode
        ):
            uid, iid = id2index[user], id2index[item]
            edge.append([uid, iid])
            label.append(acode)

        edge = torch.LongTensor(edge).T
        label = torch.LongTensor(label)
        return dict(edge=edge.to(device), label=label.to(device))

    def print_data_stat(data: pd.DataFrame, name: str) -> None:
        userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
        n_user, n_item = len(userid), len(itemid)

        logger.info(f"{name} Dataset Info")
        logger.info(f" * Num. Users    : {n_user}")
        logger.info(f" * Max. UserID   : {max(userid)}")
        logger.info(f" * Num. Items    : {n_item}")
        logger.info(f" * Num. Records  : {len(data)}")

    def split_validation(self):
        return None
