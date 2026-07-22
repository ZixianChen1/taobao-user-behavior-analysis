"""
最终模型分析。

作用：
1. 使用最佳模型 XGBoost 训练
2. 保存模型文件
3. 统计模型大小
4. 测试预测速度
"""


import sys
import time
from pathlib import Path

import pandas as pd
import joblib

from xgboost import XGBClassifier


# 项目路径

BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))


PARQUET_DIR = BASE_PATH / "output_parquet"

MODEL_DIR = BASE_PATH / "model_output"

MODEL_DIR.mkdir(exist_ok=True)



LABEL_COLUMN = "is_buy"



TRAIN_FILE = "07_train_dataset.parquet"

TEST_FILE = "07_test_dataset.parquet"



RANDOM_STATE = 42



def load_data(file):

    """
    读取Parquet数据。
    """

    return pd.read_parquet(
        PARQUET_DIR / file
    )



def train_final_model(
    x_train,
    y_train
):

    """
    训练最终XGBoost模型。
    """


    positive = (
        y_train == 1
    ).sum()

    negative = (
        y_train == 0
    ).sum()


    scale_pos_weight = (
        negative / positive
    )


    model = XGBClassifier(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        scale_pos_weight=
        scale_pos_weight,

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
    保存模型文件。
    """

    model_path = (
        MODEL_DIR /
        "xgboost_final_model.pkl"
    )


    joblib.dump(
        model,
        model_path
    )


    size_mb = (
        model_path.stat().st_size
        /
        1024
        /
        1024
    )


    result = pd.DataFrame({

        "model":[
            "XGBoost"
        ],

        "model_file":[
            str(model_path)
        ],

        "size_MB":[
            round(size_mb,3)
        ]

    })


    result.to_csv(

        MODEL_DIR /
        "model_size_summary.csv",

        index=False,

        encoding="utf-8-sig"

    )


    return size_mb



def prediction_speed_test(
    model,
    x_test
):

    """
    测试预测速度。
    """


    sample = x_test.head(
        10000
    )


    start = time.time()


    model.predict_proba(
        sample
    )


    end = time.time()


    total_time = (
        end-start
    )


    result = pd.DataFrame({

        "sample_count":[
            len(sample)
        ],

        "prediction_time_seconds":[
            round(total_time,4)
        ],

        "avg_time_per_sample_ms":[

            round(
                total_time
                /
                len(sample)
                *
                1000,

                6
            )

        ]

    })


    result.to_csv(

        MODEL_DIR /
        "prediction_speed_summary.csv",

        index=False,

        encoding="utf-8-sig"

    )


    return result



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



    print(
        "训练最终模型..."
    )


    model = train_final_model(
        x_train,
        y_train
    )


    size = save_model(
        model
    )


    speed = prediction_speed_test(
        model,
        x_test
    )


    print(
        "模型大小(MB):",
        round(size,3)
    )


    print(
        speed
    )


    print(
        "最终模型分析完成"
    )



if __name__ == "__main__":

    main()