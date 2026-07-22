"""
第九阶段补充实验：class_weight 平衡方法。

作用：
1. 使用原始训练集
2. 在模型训练阶段加入类别权重
3. 与不同采样方法进行对比
4. 保存实验结果
"""


import sys
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


# 项目根目录
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))


OUTPUT_DIR = BASE_PATH / "output_parquet"


LABEL_COLUMN = "is_buy"


TRAIN_FILE = "07_train_dataset.parquet"

VALID_FILE = "07_valid_dataset.parquet"


def load_data(file_name):
    """
    读取 Parquet 数据。
    """

    return pd.read_parquet(
        OUTPUT_DIR / file_name,
        engine="pyarrow"
    )


def evaluate(
    y_true,
    y_prob
):
    """
    计算模型指标。
    """

    y_pred = (
        y_prob >= 0.5
    ).astype(int)


    return {
        "auc":
            roc_auc_score(
                y_true,
                y_prob
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                y_prob
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
    }



def main():

    print("读取训练集...")


    train_df = load_data(
        TRAIN_FILE
    )


    valid_df = load_data(
        VALID_FILE
    )


    X_train = train_df.drop(
        columns=[LABEL_COLUMN]
    )

    y_train = train_df[LABEL_COLUMN]


    X_valid = valid_df.drop(
        columns=[LABEL_COLUMN]
    )

    y_valid = valid_df[LABEL_COLUMN]



    print("开始训练 class_weight 模型...")


    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    y_prob = model.predict_proba(
        X_valid
    )[:,1]


    result = evaluate(
        y_valid,
        y_prob
    )


    result["experiment"] = (
        "class_weight_balanced"
    )

    result["train_rows"] = len(train_df)



    result_df = pd.DataFrame(
        [result]
    )


    result_df.to_parquet(
        OUTPUT_DIR /
        "09_class_weight_result.parquet",
        index=False,
        engine="pyarrow"
    )


    print(
        "class_weight实验完成"
    )

    print(result_df)



if __name__ == "__main__":
    main()