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
                learning_rate=self.args.learning_rate,
                iterations=self.args.iterations,
                task_type="GPU",
            )
        elif args.model == "XG":
            self.model = xgb.XGBClassifier(
                learning_rate=self.args.learning_rate,
                n_estimators=self.args.iterations,
                max_depth=self.args.max_depth,
            )
        elif args.model == "LGBM":
            self.model = lgbm.LGBMClassifier(
                learning_rate=self.args.learning_rate,
                n_estimators=self.args.iterations,
                max_depth=self.args.max_depth,
            )
        else:
            raise Exception("cat,xg,lgbm 중 하나의 모델을 선택해주세요")

    def training(self, data):
        print("###start MODEL training ###")
        self.model.fit(
            self.train[self.feature],
            self.train_value,
            early_stopping_rounds=100,
            cat_features=list(self.train[FEATURE]),
            verbose=500,
        )

    def inference(self):
        # submission 제출하기 위한 코드
        print("### Inference && Save###")
        test_pred = self.model.predict_proba(self.test[self.feature])[:, 1]
        self.test["prediction"] = _test_pred
        submission = self.test["prediction"].reset_index(drop=True).reset_index()
        submission.rename(columns={"index": "id"}, inplace=True)
        submission.to_csv(
            os.path.join(self.args.output_path, "submission.csv"), index=False
        )
