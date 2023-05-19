import pandas as pd
import numpy as np
import os

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgbm

# import optuna
# from optuna import Trial, visualization
# from optuna.samplers import TPESampler


class boosting_model:
    def __init__(self, args, FEATURE):
        self.args = args
        self.feature = FEATURE
        if args.model == "CAT":
            self.model = CatBoostClassifier(
                learning_rate=float(self.args.learning_rate),
                iterations=int(self.args.iterations),
                task_type="GPU",
            )
        elif args.model == "XG":
            self.model = xgb.XGBClassifier(
                learning_rate=float(self.args.learning_rate),
                # n_estimators=int(self.args.iterations),
                # max_depth=self.args.max_depth,
            )
        elif args.model == "LGBM":
            self.model = lgbm.LGBMClassifier(
                learning_rate=int(self.args.learning_rate),
                # n_estimators=self.args.iterations,
                # max_depth=self.args.max_depth,
            )
        else:
            raise Exception("cat,xg,lgbm 중 하나의 모델을 선택해주세요")

    def training(self, data, args):
        print("###start MODEL training ###")
        if args.model == "CAT":
            print(self.feature)
            print(data["train_x"][self.feature])
            self.model.fit(
                data["train_x"][self.feature],
                data["train_y"],
                early_stopping_rounds=100,
                cat_features=list(data["train_x"][self.feature]),
                verbose=500,
            )
        else:
            self.model.fit(
                data["train_x"][self.feature],
                data["train_y"],
                early_stopping_rounds=100,
                verbose=500,
            )

    def inference(self, data):
        # submission 제출하기 위한 코드
        print("### Inference && Save###")
        test_pred = self.model.predict_proba(data["test"][self.feature])[:, 1]
        data["test"]["prediction"] = test_pred
        submission = data["test"]["prediction"].reset_index(drop=True).reset_index()
        submission.rename(columns={"index": "id"}, inplace=True)
        submission.to_csv(
            os.path.join(self.args.output_dir, "submission.csv"), index=False
        )
