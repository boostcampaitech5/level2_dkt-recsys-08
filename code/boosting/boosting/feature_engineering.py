import numpy as np
import pandas as pd
from tqdm import tqdm


def percentile(s):
    return np.sum(s) / len(s)


def elo(df, feature, elo_feat):
    def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return theta + learning_rate_theta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return beta - learning_rate_beta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def learning_rate_theta(nb_answers):
        return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

    def learning_rate_beta(nb_answers):
        return 1 / (1 + 0.05 * nb_answers)

    def probability_of_good_answer(theta, beta, left_asymptote):
        return left_asymptote + (1 - left_asymptote) + sigmoid(theta - beta)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def estimate_parameters(answers_df, granularity_feature_name=feature):
        item_parameters = {
            granularity_feature_value: {"beta": 0, "nb_answers": 0}
            for granularity_feature_value in np.unique(
                answers_df[granularity_feature_name]
            )
        }
        student_parameters = {
            student_id: {"theta": 0, "nb_answers": 0}
            for student_id in np.unique(answers_df.userID)
        }

        for student_id, item_id, left_asymptote, answered_correctly in zip(
            answers_df.userID.values,
            answers_df[granularity_feature_name].values,
            answers_df.left_asymptote.values,
            answers_df.answerCode.values,
        ):
            theta = student_parameters[student_id]["theta"]
            beta = item_parameters[item_id]["beta"]

            item_parameters[item_id]["beta"] = get_new_beta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                item_parameters[item_id]["nb_answers"],
            )
            student_parameters[student_id]["theta"] = get_new_theta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                student_parameters[student_id]["nb_answers"],
            )

            item_parameters[item_id]["nb_answers"] += 1
            student_parameters[student_id]["nb_answers"] += 1

        return student_parameters, item_parameters

    def gou_func(theta, beta):
        return 1 / (1 + np.exp(-(theta - beta)))

    new_df = df.copy()
    new_df["left_asymptote"] = 0
    student_parameters, item_parameters = estimate_parameters(new_df)

    prob = [
        gou_func(student_parameters[student]["theta"], item_parameters[item]["beta"])
        for student, item in zip(new_df.userID.values, new_df[feature].values)
    ]

    new_df[elo_feat] = prob
    col_list = list(new_df.columns[6:-1])
    new_df.drop(columns=col_list, inplace=True)

    return new_df


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
            .agg("median")
            .reset_index()
        )
        df_time.rename(columns={"elapsed": "user_median_elapsed"}, inplace=True)
        df_time = df.merge(df_time, on="userID", how="left")
        df_time["timeDelta_userAverage"] = (
            df_time["elapsed"] - df_time["user_median_elapsed"]
        )
        return df_time["timeDelta_userAverage"]

        cumcount = (
            df.loc[:, ["userID", "answerCode"]]
            .groupby("userID")
            .agg({"answerCode": "cumcount"})
            + 1
        )
        cum_ans = cumsum / cumcount
        return cum_ans

    df["elapsed"] = elapsed_time(df)
    df["timeDelta_userAverage"] = timeDelta_from_user_average(df)
    # 대분류 feature 추가
    df["category_high"] = df["testId"].apply(lambda x: x[2])
    # 문항 순서 Feature 추가
    df["problem_num"] = df["assessmentItemID"].apply(lambda x: x[-3:])
    # 시간 관련 요소 추가
    df["hour"] = df["Timestamp"].dt.hour
    df["weekofyear"] = df["Timestamp"].dt.isocalendar().week

    # 유저별 누적 푼 문제수, 누적 맞춘 문제 갯수, 누적 정답률, 누적 푼 문제시간
    df["problem_correct_per_user"] = (
        df.loc[:, ["userID", "answerCode"]]
        .groupby("userID")
        .agg({"answerCode": "cumsum"})
    )
    df["problem_solved_per_user"] = (
        df.loc[:, ["userID", "answerCode"]]
        .groupby("userID")
        .agg({"answerCode": "cumcount"})
        + 1
    )
    df["cum_answerRate_per_user"] = (
        df["problem_correct_per_user"] / df["problem_solved_per_user"]
    )

    df["acc_elapsed_per_user"] = (
        df.loc[:, ["userID", "elapsed"]].groupby("userID").agg({"elapsed": "cumsum"})
    )

    # 대분류별 누적 풀린 횟수, 대분류별 누적 정답수, 대분류별 누적 정답률
    df["correct_answer_per_cat"] = (
        df.groupby(["userID", "category_high"])["answerCode"]
        .transform(lambda x: x.cumsum().shift(1))
        .fillna(0)
    )
    df["acc_count_per_cat"] = (
        df.loc[:, ["category_high", "answerCode"]]
        .groupby("category_high")
        .agg({"answerCode": "cumcount"})
    )
    df["acc_answerRate_per_cat"] = (
        df["correct_answer_per_cat"] / df["acc_count_per_cat"]
    )

    # week of year별 누적 풀린 횟수, 누적 정답수, 누적 정답률
    # week of year
    df["problem_correct_per_woy"] = (
        df.loc[:, ["weekofyear", "answerCode"]]
        .groupby("weekofyear")
        .agg({"answerCode": "cumsum"})
    )
    df["problem_solved_per_woy"] = (
        df.loc[:, ["weekofyear", "answerCode"]]
        .groupby("weekofyear")
        .agg({"answerCode": "cumcount"})
        + 1
    )
    df["cum_answerRate_per_woy"] = (
        df["problem_correct_per_woy"] / df["problem_solved_per_woy"]
    )

    # 이 전에 정답을 맞췄는지로 시간적 요소 반영
    df["timestep_1"] = df.groupby("userID")["answerCode"].shift(1).fillna(1).astype(int)
    df["timestep_2"] = df.groupby("userID")["answerCode"].shift(2).fillna(1).astype(int)
    df["timestep_3"] = df.groupby("userID")["answerCode"].shift(3).fillna(1).astype(int)
    df["timestep_4"] = df.groupby("userID")["answerCode"].shift(4).fillna(1).astype(int)
    df["timestep_5"] = df.groupby("userID")["answerCode"].shift(5).fillna(1).astype(int)
    # 문제 난이도 계산
    df["elo_assessment"] = elo(df, "assessmentItemID", "elo_assessment")[
        "elo_assessment"
    ]
    # 시험지 난이도 계산
    df["elo_test"] = elo(df, "testId", "elo_test")["elo_test"]
    # 태그 난이도 계산
    df["elo_tag"] = elo(df, "KnowledgeTag", "elo_tag")["elo_tag"]
    return df


class FE_aggregation:
    def __init__(self):
        pass
        # self.df = df

    ##### 유저 단위
    def feature_per_user(self, df: pd.DataFrame) -> pd.DataFrame:
        # 유저별 정답률, 정답 맞춘 횟수, 평균 소요시간, 푼 문제수
        tem1 = df.groupby("userID")["answerCode"]
        tem1 = pd.DataFrame(
            {"answerRate_per_user": tem1.mean(), "answer_cnt_per_user": tem1.count()}
        ).reset_index()
        tem2 = df.groupby("userID")["elapsed"]
        tem2 = pd.DataFrame(
            {"elapsed_time_median_per_user": tem2.median()}
        ).reset_index()
        df_user = pd.merge(tem1, tem2, on=["userID"], how="left")

        tem3 = df.groupby("userID").agg({"assessmentItemID": "count"})
        df_user = pd.merge(df_user, tem3, on=["userID"], how="left")
        df_user.rename(
            columns={"assessmentItemID": "assessment_solved_per_user"}, inplace=True
        )
        return df_user

    ##### 문제 단위
    def feature_per_item(self, df: pd.DataFrame):
        def median_elapsed_per_assessmentID(
            df: pd.DataFrame, answerCode: int
        ) -> pd.DataFrame:
            """
            해당 문제에 대한 정답자들의 평균 문제 풀이 시간, 오답자들의 평균 문제 풀이 시간
            Input
            df : train or test data
            answerCode: 정답 / 오답 여부

            Output
            df_median_elapsed : 해당 문제 정답/오답자의 풀이 시간 중위수

            """
            col_name = ["wrong_users_median_elapsed", "correct_users_median_elapsed"]
            df_median_elapsed = (
                df.loc[:, ["assessmentItemID", "answerCode", "elapsed"]]
                .groupby(["assessmentItemID", "answerCode"])
                .agg("median")
                .reset_index()
            )
            df_median_elapsed = df_median_elapsed[
                df_median_elapsed["answerCode"] == answerCode
            ].drop("answerCode", axis=1)
            df_median_elapsed.rename(
                columns={"elapsed": col_name[answerCode]}, inplace=True
            )
            return df_median_elapsed

        # 문제별 정답률, 정답 맞춘 횟수, 소요시간 median
        tem1 = df.groupby("assessmentItemID")["answerCode"]
        tem1 = pd.DataFrame(
            {"answerRate_per_item": tem1.mean(), "answer_cnt_per_item": tem1.count()}
        ).reset_index()
        tem2 = df.groupby("assessmentItemID")["elapsed"]
        tem2 = pd.DataFrame(
            {"elapsed_time_median_per_item": tem2.median()}
        ).reset_index()
        df_item = pd.merge(tem1, tem2, on=["assessmentItemID"], how="left")

        # 해당 문제에 대한 정답자들의 평균 문제 풀이 시간, 오답자들의 평균 문제 풀이 시간
        tem3 = median_elapsed_per_assessmentID(df, 0)
        df_item = pd.merge(df_item, tem3, on=["assessmentItemID"], how="left")
        tem4 = median_elapsed_per_assessmentID(df, 1)
        df_item = pd.merge(df_item, tem4, on=["assessmentItemID"], how="left")
        return df_item

    # tag 단위
    def feature_per_tag(self, df: pd.DataFrame):
        # tag가 풀린 횟수, tag별 정답률
        tem1 = (
            df.groupby("KnowledgeTag")
            .agg({"userID": "count", "answerCode": percentile})
            .reset_index()
        )
        tem1.rename(
            columns={"userID": "tag_exposed", "answerCode": "answerRate_per_tag"},
            inplace=True,
        )

        return tem1

    # 시험지(testId) 단위
    def feature_per_test(self, df: pd.DataFrame):
        # 시험지별 문제 푼 시간의 중위수, 정답률
        tem1 = (
            df.groupby("testId")
            .agg({"elapsed": "median", "answerCode": percentile})
            .reset_index()
        )
        tem1.rename(
            columns={
                "elapsed": "elapsed_median_per_test",
                "answerCode": "answerRate_per_test",
            },
            inplace=True,
        )
        return tem1

    # category 단위
    def feature_per_category(self, df: pd.DataFrame):
        # category별 문제 푼 시간의 중위수, 정답률
        tem1 = (
            df.groupby("category_high")
            .agg({"elapsed": "median", "answerCode": percentile})
            .reset_index()
        )
        tem1.rename(
            columns={
                "elapsed": "elapsed_median_per_cat",
                "answerCode": "answerRate_per_cat",
            },
            inplace=True,
        )
        return tem1

    # item num 단위
    def feature_per_problem_num(self, df: pd.DataFrame):
        # 문항 순서별 문제 푼 시간의 중위수, 정답률
        tem1 = (
            df.groupby("problem_num")
            .agg({"elapsed": "median", "answerCode": percentile})
            .reset_index()
        )
        tem1.rename(
            columns={
                "elapsed": "elapsed_median_per_problem_num",
                "answerCode": "answerRate_per_problem_num",
            },
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
    df_test = fe.feature_per_test(df_1)
    df_cat = fe.feature_per_category(df_1)
    df_problem_num = fe.feature_per_problem_num(df_1)

    df = df_2.merge(df_user, how="left", on="userID")
    df = df.merge(df_assessment, how="left", on="assessmentItemID")
    df = df.merge(df_tag, how="left", on="KnowledgeTag")
    df = df.merge(df_test, how="left", on="testId")
    df = df.merge(df_cat, how="left", on="category_high")
    df = df.merge(df_problem_num, how="left", on="problem_num")
    return df
