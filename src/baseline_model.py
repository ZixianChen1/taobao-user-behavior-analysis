"""
第九阶段：Baseline 模型训练。

作用：
1. 读取不同样本平衡方案训练集
2. 使用 Logistic Regression 训练模型
3. 在统一验证集上预测
4. 输出不同方案评估指标
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


BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))


PARQUET_DIR = BASE_PATH / "output_parquet"


LABEL_COLUMN = "is_buy"


TRAIN_FILES = {
    "original": "07_train_dataset.parquet",
    "under_sample_1_10": "08_train_under_sample_1_to_10.parquet",
    "under_sample_1_5": "08_train_under_sample_1_to_5.parquet",
    "under_sample_1_3": "08_train_under_sample_1_to_3.parquet"
}


VALID_FILE = "07_valid_dataset.parquet"


RANDOM_STATE = 42


def load_parquet(file_name):
    """
    读取 Parquet 数据。
    """
    return pd.read_parquet(
        PARQUET_DIR / file_name,
        engine="pyarrow"
    )


def evaluate_model(
    y_true,
    y_prob
):
    """
    计算模型评估指标。
    """

    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(
            y_true,
            y_prob
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_prob
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }


def train_one_experiment(
    name,
    train_file,
    valid_df
):
    """
    训练一个样本平衡方案。
    """

    train_df = load_parquet(train_file)


    X_train = train_df.drop(
        columns=[LABEL_COLUMN]
    )

    y_train = train_df[LABEL_COLUMN]


    X_valid = valid_df.drop(
        columns=[LABEL_COLUMN]
    )

    y_valid = valid_df[LABEL_COLUMN]


    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )


    model.fit(
        X_train,
        y_train
    )


    y_prob = model.predict_proba(
        X_valid
    )[:, 1]


    metrics = evaluate_model(
        y_valid,
        y_prob
    )


    metrics["experiment"] = name

    metrics["train_rows"] = len(train_df)


    return metrics



def main():

    valid_df = load_parquet(
        VALID_FILE
    )


    results = []


    for name, file in TRAIN_FILES.items():

        print(
            f"开始训练：{name}"
        )


        result = train_one_experiment(
            name,
            file,
            valid_df
        )


        results.append(
            result
        )


    result_df = pd.DataFrame(results)


    result_df = result_df[
        [
            "experiment",
            "train_rows",
            "auc",
            "pr_auc",
            "precision",
            "recall",
            "f1"
        ]
    ]


    result_df.to_parquet(
        PARQUET_DIR / "09_baseline_results.parquet",
        index=False,
        engine="pyarrow"
    )


    print(
        "Baseline训练完成"
    )

    print(result_df)



if __name__ == "__main__":
    main()