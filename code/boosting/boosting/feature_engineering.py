import numpy as np
import pandas as pd


class FE:
    def __init__(self):
        pass
        # self.df = df

    def elapsed_time(self, df: pd.DataFrame) -> pd.Series:
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

    ##### 유저 단위
    def feature_per_user(self, df: pd.DataFrame) -> pd.DataFrame:
        #         def timeDelta_from_user_average(df: pd.DataFrame) -> pd.Series:
        #             """
        #             해당 문제 풀이 시간 - 해당 유저 평균 문제 풀이 시간

        #             Input
        #             df: train or test data

        #             Output
        #             df_time['timeDelta_userAverage'] : problem-solving time deviation from user average

        #             """
        #             df_time = (
        #                 df.loc[:, ["userID", "elapsed"]]
        #                 .groupby(["userID"])
        #                 .agg("mean")
        #                 .reset_index()
        #             )
        #             df_time.rename(columns={"elapsed": "user_mean_elapsed"}, inplace=True)
        #             df_time = df.merge(df_time, on="userID", how="left")
        #             df_time["timeDelta_userAverage"] = (
        #                 df_time["elapsed"] - df_time["user_mean_elapsed"]
        #             )
        #             return df_time["timeDelta_userAverage"]
        #         df_temp = df[['userID','assessmentItemID']]
        #         df_temp['timeDelta_userAverage'] = timeDelta_from_user_average(df)

        # 유저별 정답룰, 정답 맞춘 횟수, 평균 소요시간
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
        #             def mean_elapsed_per_assessmentID_correct_or_not(
        #                 df: pd.DataFrame,
        #             ) -> pd.DataFrame:
        #                 """
        #                 해당 문제에 대한 정답자들의 평균 문제 풀이 시간, 오답자들의 평균 문제 풀이 시간
        #                 Input
        #                 df : train or test data
        #                 answerCode: 정답 / 오답 여부

        #                 Output
        #                 df_mean_elapsed_per_assessmentID : 해당 문제 정답/오답자의 풀이 시간 평균

        #                 """

        #                 def mean_elapsed_per_assessmentID(
        #                     df: pd.DataFrame, answerCode: int
        #                 ) -> pd.Series:
        #                     col_name = ["wrong_users_mean_elapsed", "correct_users_mean_elapsed"]
        #                     df_assessment = (
        #                         df.loc[:, ["assessmentItemID", "answerCode", "elapsed"]]
        #                         .groupby(["assessmentItemID", "answerCode"])
        #                         .agg("mean")
        #                         .reset_index()
        #                     )
        #                     df_assessment = df_assessment[
        #                         df_assessment["answerCode"] == answerCode
        #                     ].drop("answerCode", axis=1)
        #                     df_assessment.rename(
        #                         columns={"elapsed": col_name[answerCode]}, inplace=True
        #                     )
        #                     df_mean_elapsed_per_assessmentID = df.loc[
        #                         :, ["assessmentItemID"]
        #                     ].merge(df_assessment, on="assessmentItemID", how="left")[
        #                         [col_name[answerCode]]
        #                     ]

        #                     return df_mean_elapsed_per_assessmentID

        #                 df_correct_wrong = pd.DataFrame()
        #                 df_correct_wrong["assessmentItemID"] = df["assessmentItemID"]
        #                 df_correct_wrong[
        #                     "wrong_users_mean_elapsed"
        #                 ] = mean_elapsed_per_assessmentID(df, 0)
        #                 df_correct_wrong[
        #                     "correct_users_mean_elapsed"
        #                 ] = mean_elapsed_per_assessmentID(df, 1)
        #                 df["wrong_users_mean_elapsed"] = df_correct_wrong[
        #                     "wrong_users_mean_elapsed"
        #                 ]
        #                 df["correct_users_mean_elapsed"] = df_correct_wrong[
        #                     "correct_users_mean_elapsed"
        #                 ]
        #                 df_correct_wrong = df_correct_wrong.drop_duplicates()
        #                 df_users_mean_elapsed = df.merge(
        #                     df_correct_wrong, on="assessmentItemID", how="left"
        #                 )[["wrong_users_mean_elapsed", "correct_users_mean_elapsed"]]
        #                 print(df_users_mean_elapsed.columns)
        #                 return df_users_mean_elapsed
        #             df_new_col = mean_elapsed_per_assessmentID_correct_or_not(df)
        #             df_temp = df[['userID','assessmentItemID']]
        #             df_temp["wrong_users_mean_elapsed"] = df_new_col['wrong_users_mean_elapsed']
        #             df_temp['correct_users_mean_elapsed'] = df_new_col['correct_users_mean_elapsed']
        #             return df_temp

        # 문제별 정답룰, 정답 맞춘 횟수, 평균 소요시간
        tem1 = df.groupby("assessmentItemID")["answerCode"]
        tem1 = pd.DataFrame(
            {"item_answer_mean": tem1.mean(), "item_answer_cnt": tem1.count()}
        ).reset_index()
        tem2 = df.groupby("assessmentItemID")["elapsed"]
        tem2 = pd.DataFrame({"item_time_mean": tem2.mean()}).reset_index()
        df_item = pd.merge(tem1, tem2, on=["assessmentItemID"], how="left")
        return df_item


def final_feature_engineering(
    df_1: pd.DataFrame, df_2: pd.DataFrame, is_train: bool
) -> pd.DataFrame:
    """
    위에 작성한 함수 호출하고 최종적으로 완성된 data 반환
    df_1 : 기준점이 되는 DataFrame
    """
    fe = FE()
    df_1["elapsed"] = fe.elapsed_time(df_1)
    df_user = fe.feature_per_user(df_1)
    df_assessment = fe.feature_per_item(df_1)

    if is_train:
        df = df_1.merge(df_user, how="left", on="userID")
        df = df.merge(df_assessment, how="left", on="assessmentItemID")
    else:
        # elapsed를 계산할 수가 없군요
        df = df_2.merge(df_user, how="left", on="userID")
        df = df.merge(df_assessment, how="left", on="assessmentItemID")
    return df
