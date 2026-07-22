"""
第十阶段：模型对比实验。

作用：
1. 读取时间划分后的训练集、验证集、测试集
2. 使用相同特征训练不同模型
3. 对比 Logistic Regression、LightGBM、XGBoost
4. 保存模型评估结果
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

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# 项目根目录
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))


# 模型数据统一读取 Parquet
PARQUET_DIR = BASE_PATH / "output_parquet"


LABEL_COLUMN = "is_buy"


TRAIN_FILE = "07_train_dataset.parquet"
VALID_FILE = "07_valid_dataset.parquet"
TEST_FILE = "07_test_dataset.parquet"


RANDOM_STATE = 42


def load_dataset(file_name: str) -> pd.DataFrame:
    """
    读取 Parquet 数据集。
    """

    file_path = PARQUET_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"文件不存在: {file_path}"
        )

    return pd.read_parquet(
        file_path,
        engine="pyarrow"
    )


def evaluate_model(
    model_name,
    model,
    x_train,
    y_train,
    x_test,
    y_test
):
    """
    训练模型并计算评估指标。
    """

    print(f"开始训练：{model_name}")

    model.fit(
        x_train,
        y_train
    )


    # 输出购买概率
    y_prob = model.predict_proba(
        x_test
    )[:, 1]


    # 默认0.5阈值
    y_pred = (
        y_prob >= 0.5
    ).astype(int)


    result = {

        "model": model_name,

        "auc": roc_auc_score(
            y_test,
            y_prob
        ),

        "pr_auc": average_precision_score(
            y_test,
            y_prob
        ),

        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        )
    }


    print(
        f"{model_name}完成:",
        result
    )


    return result



def main():

    print("读取数据...")


    train_df = load_dataset(
        TRAIN_FILE
    )


    test_df = load_dataset(
        TEST_FILE
    )


    print(
        "训练集:",
        train_df.shape
    )

    print(
        "测试集:",
        test_df.shape
    )


    # 特征
    x_train = train_df.drop(
        columns=[LABEL_COLUMN]
    )

    y_train = train_df[LABEL_COLUMN]


    x_test = test_df.drop(
        columns=[LABEL_COLUMN]
    )

    y_test = test_df[LABEL_COLUMN]



    results = []


    # =====================
    # 1. Logistic Regression
    # =====================

    lr_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )


    results.append(
        evaluate_model(
            "LogisticRegression_class_weight",
            lr_model,
            x_train,
            y_train,
            x_test,
            y_test
        )
    )



    # =====================
    # 2. LightGBM
    # =====================

    lgb_model = LGBMClassifier(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


    results.append(
        evaluate_model(
            "LightGBM_class_weight",
            lgb_model,
            x_train,
            y_train,
            x_test,
            y_test
        )
    )



    # =====================
    # 3. XGBoost
    # =====================


    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()


    scale_pos_weight = (
        negative_count /
        positive_count
    )


    xgb_model = XGBClassifier(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,

        eval_metric="logloss",

        n_jobs=-1
    )


    results.append(
        evaluate_model(
            "XGBoost_scale_pos_weight",
            xgb_model,
            x_train,
            y_train,
            x_test,
            y_test
        )
    )



    # 保存结果

    result_df = pd.DataFrame(
        results
    )


    result_df.to_parquet(
        PARQUET_DIR /
        "10_model_comparison_results.parquet",

        index=False,

        engine="pyarrow"
    )


    result_df.to_csv(
        PARQUET_DIR /
        "10_model_comparison_results.csv",

        index=False,

        encoding="utf-8-sig"
    )


    print("\n模型对比完成")

    print(result_df)



if __name__ == "__main__":

    main()