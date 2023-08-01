from base import BaseDataLoader
import os
import time
import torch
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from base import BaseDataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from utils import get_logger, logging_conf

logger = get_logger(logging_conf)


class DKTDataLoader(BaseDataLoader):
    def __init__(self, args, training=True):
        super().__init__()
        self.data = self.load_data_from_file(args, training)
        self.data = self.feature_engineering(args, self.data, training)
        self.data = self.preprocessing(args, self.data, training)

        args.dim_cats = [
            len(np.load(os.path.join(args.asset_dir, f"{col}_classes.npy")))
            for col in args.cat_cols
        ]

    def split_data(self, args):
        return train_test_split(
            self.data, test_size=args.split_ratio, random_state=args.seed, shuffle=True
        )

    def save_labels(self, args, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def preprocessing(
        self, args, df: pd.DataFrame, is_train: bool = True
    ) -> pd.DataFrame:
        if not os.path.exists(args.asset_dir):
            os.makedirs(args.asset_dir)

        for col in args.cat_cols:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            if is_train:
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.save_labels(args, le, col)
            else:
                label_path = os.path.join(args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else "unknown")

            df[col] = le.transform(df[col])

        columns = args.cat_cols + args.num_cols + args.tgt_col
        group = df.groupby("userID").apply(
            lambda r: [r[column].values for column in columns]
        )
        return group.values

    def feature_engineering(self, args, df, training):
        if training:
            data_dir = os.path.join(args.data_dir, args.test_file_name)
            df_test = pd.read_csv(data_dir)
            df_test = df_test[df_test["answerCode"] != -1]
            df = pd.concat([df, df_test])

            userID_sliding = list()
            for key, value in df["userID"].value_counts().sort_index().items():
                q, r = divmod(value, args.max_seq_len)
                for i in range(q):
                    userID_sliding += [f"{key}a{i}"] * args.max_seq_len
                userID_sliding += [f"{key}a{i+1}"] * r
            df["userID"] = userID_sliding

        def convert_time(s: str):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)

        df["beforeCorrect"] = df["answerCode"] + 1
        df["beforeCorrect"] = (
            df.groupby("userID")["beforeCorrect"].shift(1).fillna(value=0)
        )  # 0, 1, 2

        df["consumedTime"] = df.groupby("userID")["Timestamp"].diff().fillna(value=0)
        df.loc[df["testId"] != df["testId"].shift(1), "consumedTime"] = 0

        if training:
            with open(
                args.asset_dir + "Q1_assessmentItemID_consumedTime.pkl", "wb"
            ) as f:
                pickle.dump(
                    df.groupby("assessmentItemID")["consumedTime"]
                    .apply(lambda x: x.quantile(0.25))
                    .to_dict(),
                    f,
                )

            with open(
                args.asset_dir + "Q2_assessmentItemID_consumedTime.pkl", "wb"
            ) as f:
                pickle.dump(
                    df.groupby("assessmentItemID")["consumedTime"]
                    .apply(lambda x: x.quantile(0.50))
                    .to_dict(),
                    f,
                )

            with open(
                args.asset_dir + "Q3_assessmentItemID_consumedTime.pkl", "wb"
            ) as f:
                pickle.dump(
                    df.groupby("assessmentItemID")["consumedTime"]
                    .apply(lambda x: x.quantile(0.75))
                    .to_dict(),
                    f,
                )

            with open(args.asset_dir + "value_counts_assessmentItemID.pkl", "wb") as f:
                pickle.dump(df["assessmentItemID"].value_counts().to_dict(), f)

            with open(args.asset_dir + "value_counts_testId.pkl", "wb") as f:
                pickle.dump(df["testId"].value_counts().to_dict(), f)

        with open(args.asset_dir + "Q1_assessmentItemID_consumedTime.pkl", "rb") as f:
            Q1 = pickle.load(f)
        with open(args.asset_dir + "Q2_assessmentItemID_consumedTime.pkl", "rb") as f:
            Q2 = pickle.load(f)
        with open(args.asset_dir + "Q3_assessmentItemID_consumedTime.pkl", "rb") as f:
            Q3 = pickle.load(f)
        df["consumedTime"] = df[["assessmentItemID", "consumedTime"]].apply(
            lambda x: Q2[x[0]]
            if x[1] > Q3[x[0]] + 1.5 * (Q3[x[0]] - Q1[x[0]])
            else x[1],
            axis=1,
        )

        with open(args.asset_dir + "value_counts_assessmentItemID.pkl", "rb") as f:
            df["value_counts_assessmentItemID"] = df["assessmentItemID"].map(
                pickle.load(f)
            )

        with open(args.asset_dir + "value_counts_testId.pkl", "rb") as f:
            df["value_counts_testId"] = df["testId"].map(pickle.load(f))

        if training:
            with open(
                args.asset_dir + "mean_assessmentItemID_answerCode.pkl", "wb"
            ) as f:
                pickle.dump(
                    df.groupby("assessmentItemID")["answerCode"].mean().to_dict(), f
                )

            with open(args.asset_dir + "mean_testId_answerCode.pkl", "wb") as f:
                pickle.dump(df.groupby("testId")["answerCode"].mean().to_dict(), f)

            with open(args.asset_dir + "mean_KnowledgeTag_answerCode.pkl", "wb") as f:
                pickle.dump(
                    df.groupby("KnowledgeTag")["answerCode"].mean().to_dict(), f
                )

            scaler_consumedTime = StandardScaler()
            scaler_consumedTime.fit(df[["consumedTime"]])
            with open(args.asset_dir + "scaler_consumedTime.pkl", "wb") as f:
                pickle.dump(scaler_consumedTime, f)

            scaler_value_counts_assessmentItemID = StandardScaler()
            scaler_value_counts_assessmentItemID.fit(
                df[["value_counts_assessmentItemID"]]
            )
            with open(
                args.asset_dir + "scaler_value_counts_assessmentItemID.pkl", "wb"
            ) as f:
                pickle.dump(scaler_value_counts_assessmentItemID, f)

            scaler_value_counts_testId = StandardScaler()
            scaler_value_counts_testId.fit(df[["value_counts_testId"]])
            with open(args.asset_dir + "scaler_value_counts_testId.pkl", "wb") as f:
                pickle.dump(scaler_value_counts_testId, f)

        with open(args.asset_dir + "mean_assessmentItemID_answerCode.pkl", "rb") as f:
            df["mean_assessmentItemID_answerCode"] = df["assessmentItemID"].map(
                pickle.load(f)
            )

        with open(args.asset_dir + "mean_testId_answerCode.pkl", "rb") as f:
            df["mean_testId_answerCode"] = df["testId"].map(pickle.load(f))

        with open(args.asset_dir + "mean_KnowledgeTag_answerCode.pkl", "rb") as f:
            df["mean_KnowledgeTag_answerCode"] = df["KnowledgeTag"].map(pickle.load(f))

        with open(args.asset_dir + "scaler_consumedTime.pkl", "rb") as f:
            scaler_consumedTime = pickle.load(f)
            df["consumedTime"] = scaler_consumedTime.transform(
                df[["consumedTime"]]
            ).squeeze()

        with open(
            args.asset_dir + "scaler_value_counts_assessmentItemID.pkl", "rb"
        ) as f:
            scaler_value_counts_assessmentItemID = pickle.load(f)
            df[
                "value_counts_assessmentItemID"
            ] = scaler_value_counts_assessmentItemID.transform(
                df[["value_counts_assessmentItemID"]]
            ).squeeze()

        with open(args.asset_dir + "scaler_value_counts_testId.pkl", "rb") as f:
            scaler_value_counts_testId = pickle.load(f)
            df["value_counts_testId"] = scaler_value_counts_testId.transform(
                df[["value_counts_testId"]]
            ).squeeze()

        if training:
            with open(
                args.asset_dir + "mean_assessmentItemID_consumedTime.pkl", "wb"
            ) as f:
                pickle.dump(
                    df.groupby("assessmentItemID")["consumedTime"].mean().to_dict(), f
                )

            with open(args.asset_dir + "mean_testId_consumedTime.pkl", "wb") as f:
                pickle.dump(df.groupby("testId")["consumedTime"].mean().to_dict(), f)

            with open(args.asset_dir + "mean_KnowledgeTag_consumedTime.pkl", "wb") as f:
                pickle.dump(
                    df.groupby("KnowledgeTag")["consumedTime"].mean().to_dict(), f
                )

        with open(args.asset_dir + "mean_assessmentItemID_consumedTime.pkl", "rb") as f:
            df["mean_assessmentItemID_consumedTime"] = df["assessmentItemID"].map(
                pickle.load(f)
            )

        with open(args.asset_dir + "mean_testId_answerCode.pkl", "rb") as f:
            df["mean_testId_answerCode"] = df["testId"].map(pickle.load(f))

        with open(args.asset_dir + "mean_KnowledgeTag_answerCode.pkl", "rb") as f:
            df["mean_KnowledgeTag_answerCode"] = df["KnowledgeTag"].map(pickle.load(f))

        return df

    def load_data_from_file(self, args, training):
        if training:
            data_dir = os.path.join(args.data_dir, args.train_file_name)
        else:
            data_dir = os.path.join(args.data_dir, args.test_file_name)
        df = pd.read_csv(data_dir)
        return df

    def get_loader(self, args, dataset, data, shuffle):
        dataset = dataset(args, data)
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=args.num_workers,
            shuffle=shuffle,
            batch_size=args.batch_size,
        )
        return loader


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, args, data: np.ndarray):
        self.data = data
        self.max_seq_len = args.max_seq_len

        self.cat_cols = args.cat_cols
        self.num_cols = args.num_cols
        self.tgt_col = args.tgt_col

        self.n_cats = len(self.cat_cols)
        self.n_nums = len(self.num_cols)

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]

        temp = list()
        for i in range(self.n_cats):
            temp.append((self.cat_cols[i], torch.tensor(row[i] + 1)))
        for i in range(self.n_cats, self.n_cats + self.n_nums):
            temp.append((self.num_cols[i - self.n_cats], torch.tensor(row[i])))
        temp.append((self.tgt_col[0], torch.tensor(row[-1])))
        data = dict(temp)

        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len :]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len - seq_len :] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask

        data = {k: v.float() for k, v in data.items()}
        return data

    def __len__(self) -> int:
        return len(self.data)
