"""
最终模型分析。

作用：
1. 使用最佳模型 XGBoost 训练
2. 保存模型文件
3. 统计模型大小
4. 随机抽样测试预测速度
"""


import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from xgboost import XGBClassifier


# 项目路径
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))


PARQUET_DIR = BASE_PATH / "output_parquet"

MODEL_DIR = BASE_PATH / "model_output"

MODEL_DIR.mkdir(
    exist_ok=True
)


LABEL_COLUMN = "is_buy"


TRAIN_FILE = "07_train_dataset.parquet"

TEST_FILE = "07_test_dataset.parquet"


RANDOM_STATE = 42

PREDICTION_SAMPLE_COUNT = 10000

PREDICTION_REPEAT_TIMES = 3


def load_data(file_name):
    """
    读取 Parquet 数据。
    """

    file_path = (
        PARQUET_DIR
        / file_name
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"文件不存在：{file_path}"
        )

    return pd.read_parquet(
        file_path,
        engine="pyarrow"
    )


def train_final_model(
    x_train,
    y_train
):
    """
    训练最终 XGBoost 模型。
    """

    positive_count = int(
        (y_train == 1).sum()
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    if positive_count == 0:
        raise ValueError(
            "训练集中没有正样本，无法计算 scale_pos_weight。"
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print(
        "scale_pos_weight：",
        round(scale_pos_weight, 4)
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        x_train,
        y_train
    )

    return model


def save_model(model):
    """
    保存模型并统计模型大小。
    """

    model_path = (
        MODEL_DIR
        / "xgboost_final_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    size_mb = (
        model_path.stat().st_size
        / 1024
        / 1024
    )

    result_df = pd.DataFrame({
        "model": [
            "XGBoost"
        ],
        "model_file": [
            str(model_path)
        ],
        "size_mb": [
            round(size_mb, 3)
        ]
    })

    result_df.to_csv(
        MODEL_DIR
        / "model_size_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return size_mb


def prediction_speed_test(
    model,
    x_test
):
    """
    随机抽样测试预测速度。
    """

    if len(x_test) == 0:
        raise ValueError(
            "测试集为空，无法测试预测速度。"
        )

    sample_count = min(
        PREDICTION_SAMPLE_COUNT,
        len(x_test)
    )

    prediction_times = []

    detail_results = []

    print(
        f"开始预测速度测试，共重复 "
        f"{PREDICTION_REPEAT_TIMES} 次。"
    )

    for repeat_index in range(
        PREDICTION_REPEAT_TIMES
    ):

        current_random_state = (
            RANDOM_STATE
            + repeat_index
        )

        # 每次从测试集中随机抽取样本
        sample = x_test.sample(
            n=sample_count,
            random_state=current_random_state,
            replace=False
        )

        start_time = time.perf_counter()

        model.predict_proba(
            sample
        )

        end_time = time.perf_counter()

        prediction_time = (
            end_time
            - start_time
        )

        avg_time_per_sample_ms = (
            prediction_time
            / sample_count
            * 1000
        )

        prediction_times.append(
            prediction_time
        )

        detail_results.append({
            "test_round": repeat_index + 1,
            "sample_method": "random_sample",
            "sample_count": sample_count,
            "random_state": current_random_state,
            "prediction_time_seconds": round(
                prediction_time,
                6
            ),
            "avg_time_per_sample_ms": round(
                avg_time_per_sample_ms,
                6
            )
        })

        print(
            f"第 {repeat_index + 1} 次测试完成，"
            f"耗时：{prediction_time:.6f} 秒"
        )

    average_prediction_time = (
        sum(prediction_times)
        / len(prediction_times)
    )

    average_time_per_sample_ms = (
        average_prediction_time
        / sample_count
        * 1000
    )

    summary_df = pd.DataFrame({
        "sample_method": [
            "random_sample"
        ],
        "sample_count": [
            sample_count
        ],
        "repeat_times": [
            PREDICTION_REPEAT_TIMES
        ],
        "average_prediction_time_seconds": [
            round(
                average_prediction_time,
                6
            )
        ],
        "average_time_per_sample_ms": [
            round(
                average_time_per_sample_ms,
                6
            )
        ],
        "minimum_prediction_time_seconds": [
            round(
                min(prediction_times),
                6
            )
        ],
        "maximum_prediction_time_seconds": [
            round(
                max(prediction_times),
                6
            )
        ]
    })

    detail_df = pd.DataFrame(
        detail_results
    )

    summary_df.to_csv(
        MODEL_DIR
        / "prediction_speed_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    detail_df.to_csv(
        MODEL_DIR
        / "prediction_speed_detail.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return summary_df, detail_df


def main():
    """
    运行最终模型分析。
    """

    print(
        "读取数据..."
    )

    train_df = load_data(
        TRAIN_FILE
    )

    test_df = load_data(
        TEST_FILE
    )

    print(
        f"训练集维度：{train_df.shape}"
    )

    print(
        f"测试集维度：{test_df.shape}"
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

    print(
        "训练最终模型..."
    )

    model = train_final_model(
        x_train,
        y_train
    )

    print(
        "保存最终模型..."
    )

    model_size = save_model(
        model
    )

    print(
        "测试预测速度..."
    )

    speed_summary, speed_detail = (
        prediction_speed_test(
            model,
            x_test
        )
    )

    print(
        "\n模型大小（MB）：",
        round(
            model_size,
            3
        )
    )

    print(
        "\n预测速度详细结果："
    )

    print(
        speed_detail
    )

    print(
        "\n预测速度平均结果："
    )

    print(
        speed_summary
    )

    print(
        "\n最终模型分析完成。"
    )


if __name__ == "__main__":
    main()