"""
第八阶段：样本不平衡处理对照实验准备。

作用：
1. 读取第七阶段生成的训练集、验证集和测试集
2. 统计原始训练集、验证集和测试集的正负样本分布
3. 只对训练集进行负样本下采样
4. 生成不同正负样本比例的训练集
5. 生成 class_weight 配置表
6. 保存样本平衡处理结果和检查结果为 Parquet
"""

import sys
from pathlib import Path

import pandas as pd


# 将项目根目录加入 Python 搜索路径，方便读取 config.py
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))


# Parquet 输出文件夹
PARQUET_DIR = BASE_PATH / "output_parquet"

LABEL_COLUMN = "is_buy"

TRAIN_FILE = "07_train_dataset.parquet"
VALID_FILE = "07_valid_dataset.parquet"
TEST_FILE = "07_test_dataset.parquet"

# 负样本下采样比例，表示 正样本 : 负样本 = 1 : ratio
NEGATIVE_SAMPLE_RATIOS = [10, 5, 3]

RANDOM_STATE = 42


def read_parquet_file(file_name: str) -> pd.DataFrame:
    """
    读取 Parquet 文件。
    """
    file_path = PARQUET_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    return pd.read_parquet(file_path, engine="pyarrow")


def save_parquet_file(df: pd.DataFrame, file_name: str) -> None:
    """
    保存 Parquet 文件。
    """
    PARQUET_DIR.mkdir(exist_ok=True)

    df.to_parquet(
        PARQUET_DIR / file_name,
        index=False,
        engine="pyarrow"
    )


def create_label_summary(
    df: pd.DataFrame,
    dataset_name: str,
    method_name: str
) -> dict:
    """
    统计一个数据集的正负样本分布。
    """
    total_count = len(df)
    buy_count = int((df[LABEL_COLUMN] == 1).sum())
    not_buy_count = int((df[LABEL_COLUMN] == 0).sum())
    buy_rate = buy_count / total_count if total_count > 0 else 0

    if buy_count > 0:
        negative_positive_ratio = not_buy_count / buy_count
    else:
        negative_positive_ratio = None

    return {
        "dataset": dataset_name,
        "method": method_name,
        "total_count": total_count,
        "not_buy_count": not_buy_count,
        "buy_count": buy_count,
        "buy_rate": buy_rate,
        "negative_positive_ratio": negative_positive_ratio
    }


def create_under_sampled_train(
    train_df: pd.DataFrame,
    negative_ratio: int
) -> pd.DataFrame:
    """
    生成负样本下采样训练集。
    """
    positive_df = train_df[train_df[LABEL_COLUMN] == 1].copy()
    negative_df = train_df[train_df[LABEL_COLUMN] == 0].copy()

    positive_count = len(positive_df)
    target_negative_count = positive_count * negative_ratio

    if target_negative_count > len(negative_df):
        target_negative_count = len(negative_df)

    sampled_negative_df = negative_df.sample(
        n=target_negative_count,
        random_state=RANDOM_STATE
    )

    balanced_train_df = pd.concat(
        [positive_df, sampled_negative_df],
        axis=0
    )

    balanced_train_df = balanced_train_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    return balanced_train_df


def create_class_weight_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成 class_weight 配置表。
    """
    total_count = len(train_df)
    positive_count = int((train_df[LABEL_COLUMN] == 1).sum())
    negative_count = int((train_df[LABEL_COLUMN] == 0).sum())

    class_0_weight = total_count / (2 * negative_count)
    class_1_weight = total_count / (2 * positive_count)

    scale_pos_weight = negative_count / positive_count

    class_weight_df = pd.DataFrame([
        {
            "method": "class_weight_balanced",
            "class_0_weight": class_0_weight,
            "class_1_weight": class_1_weight,
            "scale_pos_weight": scale_pos_weight,
            "description": "模型训练阶段使用的类别权重配置"
        }
    ])

    return class_weight_df


def create_experiment_plan() -> pd.DataFrame:
    """
    生成样本平衡实验方案表。
    """
    records = [
        {
            "experiment_name": "original_train",
            "method": "no_sampling",
            "train_file": "07_train_dataset.parquet",
            "description": "原始训练集，不做样本平衡处理"
        }
    ]

    for ratio in NEGATIVE_SAMPLE_RATIOS:
        records.append({
            "experiment_name": f"under_sample_1_to_{ratio}",
            "method": "negative_under_sampling",
            "train_file": f"08_train_under_sample_1_to_{ratio}.parquet",
            "description": f"保留全部正样本，负样本按正样本 {ratio} 倍抽样"
        })

    records.append({
        "experiment_name": "class_weight_balanced",
        "method": "class_weight",
        "train_file": "07_train_dataset.parquet",
        "description": "不改变训练集样本数量，在模型训练阶段使用类别权重"
    })

    return pd.DataFrame(records)


def run_sample_balance_experiment() -> None:
    """
    运行样本不平衡处理对照实验准备。
    """
    PARQUET_DIR.mkdir(exist_ok=True)

    train_df = read_parquet_file(TRAIN_FILE)
    valid_df = read_parquet_file(VALID_FILE)
    test_df = read_parquet_file(TEST_FILE)

    if LABEL_COLUMN not in train_df.columns:
        raise ValueError(f"训练集中不存在标签字段：{LABEL_COLUMN}")

    if LABEL_COLUMN not in valid_df.columns:
        raise ValueError(f"验证集中不存在标签字段：{LABEL_COLUMN}")

    if LABEL_COLUMN not in test_df.columns:
        raise ValueError(f"测试集中不存在标签字段：{LABEL_COLUMN}")

    balance_summary_records = []

    balance_summary_records.append(
        create_label_summary(
            train_df,
            dataset_name="train",
            method_name="original"
        )
    )

    balance_summary_records.append(
        create_label_summary(
            valid_df,
            dataset_name="valid",
            method_name="original"
        )
    )

    balance_summary_records.append(
        create_label_summary(
            test_df,
            dataset_name="test",
            method_name="original"
        )
    )

    for ratio in NEGATIVE_SAMPLE_RATIOS:
        balanced_train_df = create_under_sampled_train(
            train_df,
            negative_ratio=ratio
        )

        output_file_name = f"08_train_under_sample_1_to_{ratio}.parquet"

        save_parquet_file(
            balanced_train_df,
            output_file_name
        )

        balance_summary_records.append(
            create_label_summary(
                balanced_train_df,
                dataset_name="train",
                method_name=f"under_sample_1_to_{ratio}"
            )
        )

    balance_summary_df = pd.DataFrame(balance_summary_records)

    class_weight_df = create_class_weight_summary(train_df)

    experiment_plan_df = create_experiment_plan()

    save_parquet_file(
        balance_summary_df,
        "08_balance_summary.parquet"
    )

    save_parquet_file(
        class_weight_df,
        "08_class_weight_config.parquet"
    )

    save_parquet_file(
        experiment_plan_df,
        "08_balance_experiment_plan.parquet"
    )

    print("样本不平衡处理对照实验准备完成。")
    print(f"原始训练集行数：{len(train_df)}")
    print(f"验证集行数：{len(valid_df)}")
    print(f"测试集行数：{len(test_df)}")

    for ratio in NEGATIVE_SAMPLE_RATIOS:
        file_name = f"08_train_under_sample_1_to_{ratio}.parquet"
        print(f"已保存：{file_name}")

    print("已保存：08_balance_summary.parquet")
    print("已保存：08_class_weight_config.parquet")
    print("已保存：08_balance_experiment_plan.parquet")


if __name__ == "__main__":
    run_sample_balance_experiment()