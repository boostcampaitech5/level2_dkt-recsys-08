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
                # task_type="GPU",
                eval_metric="AUC",
            )
        elif args.model == "XG":
            self.model = xgb.XGBClassifier(
                learning_rate=self.args.learning_rate,
                n_estimators=self.args.n_estimators,
                max_depth=self.args.max_depth,
                eval_metric="AUC",
            )
        elif args.model == "LGBM":
            self.model = lgbm.LGBMClassifier(
                random_state=42,
                learning_rate=self.args.learning_rate,
                n_estimators=self.args.n_estimators,
                num_leaves=self.args.num_leaves,
            )
        else:
            raise Exception("cat,xg,lgbm 중 하나의 모델을 선택해주세요")

    def training(self, data, args):
        print("###start MODEL training ###")
        print(self.feature)
        if args.model == "CAT":
            self.model.fit(
                data["train_x"][self.feature],
                data["train_y"],
                early_stopping_rounds=50,
                cat_features=list(data["train_x"][self.feature]),
                eval_set=[(data["valid_x"][self.feature], data["valid_y"])],
                verbose=10,
            )
            print(self.model.get_best_score())
            print(self.model.get_all_params())
        else:
            self.model.fit(
                data["train_x"][self.feature],
                data["train_y"],
                early_stopping_rounds=50,
                eval_set=[(data["valid_x"][self.feature], data["valid_y"])],
                eval_metric="AUC",
                verbose=200,
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
