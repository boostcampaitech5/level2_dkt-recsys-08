import numpy as np
import pandas as pd


def percentile(s):
    return np.sum(s) / len(s)


def basic_feature_engineering(df: pd.DataFrame):
    """
    aggregation 하기 전에 전체 DataFrame에 대해 추가되는 Feature
    """

    def elapsed_time(df: pd.DataFrame) -> pd.Series:
        """
        문제 풀이 시간

        Input
        pd.DataFrame :train or test data

        Output
        pd.Series : 문제 풀이 시간(Elapsed) Feature

        """
        # 사용자와 하나의 시험지 안에서 문제 푸는데 걸린 시간
        # 같은 시험지라면 문제를 연속해서 풀었을 것으로 가정
        diff_1 = (
            df.loc[:, ["userID", "testId", "Timestamp"]]
            .groupby(["userID", "testId"])
            .diff()
            .fillna(pd.Timedelta(seconds=0))
        )
        # threshold 넘어가면 session 분리
        diff_1["elapsed"] = diff_1["Timestamp"].apply(lambda x: x.total_seconds())
        threshold = diff_1["elapsed"].quantile(0.99)
        df["session"] = diff_1["elapsed"].apply(lambda x: 0 if x < threshold else 1)
        df["session"] = (
            df.loc[:, ["userID", "testId", "session"]]
            .groupby(["userID", "testId"])
            .cumsum()
        )
        # session 나누기
        diff_2 = (
            df.loc[:, ["userID", "testId", "session", "Timestamp"]]
            .groupby(["userID", "testId", "session"])
            .diff()
            .fillna(pd.Timedelta(seconds=0))
        )
        diff_2["elapsed"] = diff_2["Timestamp"].apply(lambda x: x.total_seconds())
        df["elapsed"] = diff_2["elapsed"]
        df.drop("session", axis=1, inplace=True)
        return df["elapsed"]

    def timeDelta_from_user_average(df: pd.DataFrame) -> pd.Series:
        """
        해당 문제 풀이 시간 - 해당 유저 평균 문제 풀이 시간

        Input
        df: train or test data

        Output
        df_time['timeDelta_userAverage'] : problem-solving time deviation from user average

        """
        df_time = (
            df.loc[:, ["userID", "elapsed"]]
            .groupby(["userID"])
            .agg("mean")
            .reset_index()
        )
        df_time.rename(columns={"elapsed": "user_mean_elapsed"}, inplace=True)
        df_time = df.merge(df_time, on="userID", how="left")
        df_time["timeDelta_userAverage"] = (
            df_time["elapsed"] - df_time["user_mean_elapsed"]
        )
        return df_time["timeDelta_userAverage"]

    df["elapsed"] = elapsed_time(df)
    df["timeDelta_userAverage"] = timeDelta_from_user_average(df)

    # 이 전에 정답을 맞췄는지로 시간적 요소 반영
    df["timestep_1"] = df.groupby("userID")["answerCode"].shift(1).fillna(1).astype(int)
    df["timestep_2"] = df.groupby("userID")["answerCode"].shift(2).fillna(1).astype(int)
    df["timestep_3"] = df.groupby("userID")["answerCode"].shift(3).fillna(1).astype(int)
    df["timestep_4"] = df.groupby("userID")["answerCode"].shift(4).fillna(1).astype(int)
    df["timestep_5"] = df.groupby("userID")["answerCode"].shift(5).fillna(1).astype(int)

    # 대분류 feature 추가
    df["category_high"] = df["testId"].apply(lambda x: x[2])
    return df


class FE_aggregation:
    def __init__(self):
        pass
        # self.df = df

    ##### 유저 단위
    def feature_per_user(self, df: pd.DataFrame) -> pd.DataFrame:
        # 유저별 정답률, 정답 맞춘 횟수, 평균 소요시간
        tem1 = df.groupby("userID")["answerCode"]
        tem1 = pd.DataFrame(
            {"user_answer_mean": tem1.mean(), "user_answer_cnt": tem1.count()}
        ).reset_index()
        tem2 = df.groupby("userID")["elapsed"]
        tem2 = pd.DataFrame({"user_time_mean": tem2.mean()}).reset_index()
        df_user = pd.merge(tem1, tem2, on=["userID"], how="left")
        return df_user

    ##### 문제 단위
    def feature_per_item(self, df: pd.DataFrame):
        def mean_elapsed_per_assessmentID(
            df: pd.DataFrame, answerCode: int
        ) -> pd.DataFrame:
            """
            해당 문제에 대한 정답자들의 평균 문제 풀이 시간, 오답자들의 평균 문제 풀이 시간
            Input
            df : train or test data
            answerCode: 정답 / 오답 여부

            Output
            df_mean_elapsed : 해당 문제 정답/오답자의 풀이 시간 평균

            """
            col_name = ["wrong_users_mean_elapsed", "correct_users_mean_elapsed"]
            df_mean_elapsed = (
                df.loc[:, ["assessmentItemID", "answerCode", "elapsed"]]
                .groupby(["assessmentItemID", "answerCode"])
                .agg("mean")
                .reset_index()
            )
            df_mean_elapsed = df_mean_elapsed[
                df_mean_elapsed["answerCode"] == answerCode
            ].drop("answerCode", axis=1)
            df_mean_elapsed.rename(
                columns={"elapsed": col_name[answerCode]}, inplace=True
            )
            return df_mean_elapsed

        # 문제별 정답률, 정답 맞춘 횟수, 평균 소요시간
        tem1 = df.groupby("assessmentItemID")["answerCode"]
        tem1 = pd.DataFrame(
            {"item_answer_mean": tem1.mean(), "item_answer_cnt": tem1.count()}
        ).reset_index()
        tem2 = df.groupby("assessmentItemID")["elapsed"]
        tem2 = pd.DataFrame({"item_time_mean": tem2.mean()}).reset_index()
        df_item = pd.merge(tem1, tem2, on=["assessmentItemID"], how="left")

        # 해당 문제에 대한 정답자들의 평균 문제 풀이 시간, 오답자들의 평균 문제 풀이 시간
        tem3 = mean_elapsed_per_assessmentID(df, 0)
        df_item = pd.merge(df_item, tem3, on=["assessmentItemID"], how="left")
        tem4 = mean_elapsed_per_assessmentID(df, 1)
        df_item = pd.merge(df_item, tem4, on=["assessmentItemID"], how="left")
        return df_item

    def feature_per_tag(self, df: pd.DataFrame):
        tem1 = (
            df.groupby("KnowledgeTag")
            .agg({"userID": "count", "answerCode": percentile})
            .reset_index()
        )
        tem1.rename(
            columns={"userID": "tag_exposed", "answerCode": "tag_answer_rate"},
            inplace=True,
        )
        return tem1


def final_feature_engineering(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    """
    위에 작성한 함수 호출하고 최종적으로 완성된 data 반환
    df_1 : 기준점이 되는 DataFrame
    """
    fe = FE_aggregation()
    df_user = fe.feature_per_user(df_1)
    df_assessment = fe.feature_per_item(df_1)
    df_tag = fe.feature_per_tag(df_1)

    df = df_2.merge(df_user, how="left", on="userID")
    df = df.merge(df_assessment, how="left", on="assessmentItemID")
    df = df.merge(df_tag, how="left", on="KnowledgeTag")

    return df
