"""
学习曲线分析。

作用：
1. 使用最终XGBoost模型
2. 不同训练数据量下评估模型效果
3. 分析过拟合和欠拟合情况
4. 保存学习曲线结果
"""


import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score


BASE_PATH = Path(__file__).resolve().parent.parent

sys.path.append(
    str(BASE_PATH)
)


PARQUET_DIR = BASE_PATH / "output_parquet"


OUTPUT_DIR = BASE_PATH / "model_output"

OUTPUT_DIR.mkdir(
    exist_ok=True
)



LABEL_COLUMN = "is_buy"



TRAIN_FILE = "07_train_dataset.parquet"

TEST_FILE = "07_test_dataset.parquet"



RANDOM_STATE = 42



def load_data(file):

    return pd.read_parquet(
        PARQUET_DIR / file
    )



def train_model(
    x_train,
    y_train
):

    model = XGBClassifier(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        scale_pos_weight=
        (
            (y_train==0).sum()
            /
            (y_train==1).sum()
        ),

        eval_metric="logloss",

        random_state=RANDOM_STATE,

        n_jobs=-1

    )


    model.fit(
        x_train,
        y_train
    )


    return model



def main():

    print(
        "读取数据..."
    )


    train_df = load_data(
        TRAIN_FILE
    )


    test_df = load_data(
        TEST_FILE
    )



    x_train = train_df.drop(
        columns=[
            LABEL_COLUMN
        ]
    )


    y_train = train_df[
        LABEL_COLUMN
    ]



    x_test = test_df.drop(
        columns=[
            LABEL_COLUMN
        ]
    )


    y_test = test_df[
        LABEL_COLUMN
    ]



    # 不同训练比例

    ratios = [
        0.1,
        0.3,
        0.5,
        0.7,
        1.0
    ]


    results=[]


    for ratio in ratios:


        print(
            f"训练比例:{ratio}"
        )


        sample_size = int(
            len(x_train)*ratio
        )


        x_sample = x_train.head(
            sample_size
        )

        y_sample = y_train.head(
            sample_size
        )


        model=train_model(
            x_sample,
            y_sample
        )


        train_pred = model.predict_proba(
            x_sample
        )[:,1]


        test_pred = model.predict_proba(
            x_test
        )[:,1]



        results.append({

            "train_ratio":
            ratio,

            "train_auc":
            roc_auc_score(
                y_sample,
                train_pred
            ),

            "test_auc":
            roc_auc_score(
                y_test,
                test_pred
            )

        })



    result_df=pd.DataFrame(
        results
    )


    result_df.to_csv(

        OUTPUT_DIR /
        "learning_curve_result.csv",

        index=False,

        encoding="utf-8-sig"

    )


    # 绘图

    plt.figure(
        figsize=(8,5)
    )


    plt.plot(

        result_df["train_ratio"],

        result_df["train_auc"],

        marker="o",

        label="Train AUC"

    )


    plt.plot(

        result_df["train_ratio"],

        result_df["test_auc"],

        marker="o",

        label="Test AUC"

    )


    plt.xlabel(
        "Training Data Ratio"
    )


    plt.ylabel(
        "AUC"
    )


    plt.title(
        "XGBoost Learning Curve"
    )


    plt.legend()



    plt.savefig(

        OUTPUT_DIR /
        "learning_curve.png",

        dpi=300,

        bbox_inches="tight"

    )


    print(
        "学习曲线分析完成"
    )



if __name__=="__main__":

    main()