import numpy as np
import pandas as pd


class FE:
    def __init__(self, df: pd.DataFrame):
        self.df = df

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

    ##### 유저 단위
    def feature_per_user(df: pd.DataFrame):
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

        df["timeDelta_userAverage"] = timeDelta_from_user_average(df)
        # df에 user feature 추가해서 Return
        return df

    ##### 문제 단위
    def feature_per_item(df: pd.DataFrame):
        def mean_elapsed_per_assessmentID_correct_or_not(
            df: pd.DataFrame,
        ) -> pd.DataFrame:
            """
            해당 문제에 대한 정답자들의 평균 문제 풀이 시간, 오답자들의 평균 문제 풀이 시간
            Input
            df : train or test data
            answerCode: 정답 / 오답 여부

            Output
            df_mean_elapsed_per_assessmentID : 해당 문제 정답/오답자의 풀이 시간 평균

            """

            def mean_elapsed_per_assessmentID(
                df: pd.DataFrame, answerCode: int
            ) -> pd.Series:
                col_name = ["wrong_users_mean_elapsed", "correct_users_mean_elapsed"]
                df_assessment = (
                    df.loc[:, ["assessmentItemID", "answerCode", "elapsed"]]
                    .groupby(["assessmentItemID", "answerCode"])
                    .agg("mean")
                    .reset_index()
                )
                df_assessment = df_assessment[
                    df_assessment["answerCode"] == answerCode
                ].drop("answerCode", axis=1)
                df_assessment.rename(
                    columns={"elapsed": col_name[answerCode]}, inplace=True
                )
                df_mean_elapsed_per_assessmentID = df.loc[
                    :, ["assessmentItemID"]
                ].merge(df_assessment, on="assessmentItemID", how="left")[
                    [col_name[answerCode]]
                ]

                return df_mean_elapsed_per_assessmentID

            df_correct_wrong = pd.DataFrame()
            df_correct_wrong["assessmentItemID"] = df["assessmentItemID"]
            df_correct_wrong[
                "wrong_users_mean_elapsed"
            ] = mean_elapsed_per_assessmentID(df, 0)
            df_correct_wrong[
                "correct_users_mean_elapsed"
            ] = mean_elapsed_per_assessmentID(df, 1)
            df["wrong_users_mean_elapsed"] = df_correct_wrong[
                "wrong_users_mean_elapsed"
            ]
            df["correct_users_mean_elapsed"] = df_correct_wrong[
                "correct_users_mean_elapsed"
            ]
            df_correct_wrong = df_correct_wrong.drop_duplicates()
            df_test = df_test.merge(
                df_correct_wrong, on="assessmentItemID", how="left"
            )[["wrong_users_mean_elapsed", "correct_users_mean_elapsed"]]
            return df_test

        # df에 Item feature 추가해서 Return
        df[["wrong_users_mean_elapsed", "correct_users_mean_elapsed"]] = df_test
        return df


def add_new_feature(df):
    df["elapsed"] = elapsed_time(df)
    df_user = feature_per_user(df)
    df_assessment = feature_per_item(df)
    return df_user, df_assessment


def final_feature_engineering(
    df_1: pd.DataFrame, df_2: pd.DataFrame, is_train: bool
) -> pd.DataFrame:
    """
    위에 작성한 함수 호출하고 최종적으로 완성된 data 반환
    """
    df = FE(df_1)
    df_user, df_assessment = add_new_feature(df)
    if is_train:
        df = pd.merge(df_user, df_assessment, on=["userID", "assessmentID"])
    else:
        df = df_2.merge(df_user, how="left", on="userID")
        df = df.merge(df_assessment, how="left", on="assessmentID")
    return df
