"""
第五阶段：建模前数据检查脚本。

作用：
1. 读取第四阶段生成的特征宽表
2. 优先读取 Parquet 文件
3. 如果 Parquet 不存在，则读取 CSV 并保存为 Parquet
4. 检查宽表基本信息
5. 检查标签 is_buy 分布
6. 检查缺失值和重复 key
7. 标记 ID 字段、标签字段和疑似数据泄露字段
8. 将检查结果保存为 Parquet 文件
"""

import sys
from pathlib import Path

import pandas as pd


# 将项目根目录加入 Python 搜索路径，方便读取 config.py
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))

from config import OUTPUT_DIR


# Parquet 输出文件夹
PARQUET_DIR = BASE_PATH / "output_parquet"

WIDE_TABLE_CSV_FILE = "04_feature_wide_table.csv"
WIDE_TABLE_PARQUET_FILE = "04_feature_wide_table.parquet"

LABEL_COLUMN = "is_buy"


ID_COLUMNS = [
    "user_id",
    "item_id",
    "item_category"
]


# 这些字段和购买结果关系太直接，建模前建议排除
POSSIBLE_LEAKAGE_COLUMNS = [
    "is_buy",

    "user_item_buy_count",
    "user_item_buy_rate",

    "user_buy_count",
    "user_buy_rate",

    "item_buy_count",
    "item_buy_rate",

    "category_buy_count",
    "category_buy_rate",

    "user_category_buy_count",
    "user_category_buy_rate",

    "time_buy_count",
    "time_buy_rate",

    # total_interaction_count 包含 buy，先保守排除
    "user_item_total_interaction_count",
    "user_total_interaction_count",
    "item_total_interaction_count",
    "category_total_interaction_count",
    "user_category_total_interaction_count"
]


def read_wide_table() -> pd.DataFrame:
    """
    读取特征宽表。
    优先读取 Parquet，如果不存在则读取 CSV，并自动保存 Parquet。
    """
    PARQUET_DIR.mkdir(exist_ok=True)

    parquet_path = PARQUET_DIR / WIDE_TABLE_PARQUET_FILE
    csv_path = OUTPUT_DIR / WIDE_TABLE_CSV_FILE

    if parquet_path.exists():
        print(f"读取 Parquet 宽表：{parquet_path}")
        return pd.read_parquet(parquet_path, engine="pyarrow")

    if csv_path.exists():
        print(f"读取 CSV 宽表：{csv_path}")
        df = pd.read_csv(csv_path)

        df.to_parquet(
            parquet_path,
            index=False,
            engine="pyarrow"
        )

        print(f"已保存 Parquet 宽表：{parquet_path}")

        return df

    raise FileNotFoundError(
        f"宽表文件不存在：{parquet_path} 或 {csv_path}"
    )


def save_parquet_result(df: pd.DataFrame, file_name: str) -> None:
    """
    保存检查结果为 Parquet 文件。
    """
    parquet_path = PARQUET_DIR / f"{file_name}.parquet"

    df.to_parquet(
        parquet_path,
        index=False,
        engine="pyarrow"
    )


def create_basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成宽表基础检查结果。
    """
    total_rows = len(df)
    total_columns = df.shape[1]
    total_missing_values = int(df.isna().sum().sum())

    duplicate_key_count = int(
        df.duplicated(subset=["user_id", "item_id"]).sum()
    )

    buy_count = int((df[LABEL_COLUMN] == 1).sum())
    not_buy_count = int((df[LABEL_COLUMN] == 0).sum())
    buy_rate = buy_count / total_rows if total_rows > 0 else 0

    summary_data = [
        {
            "check_item": "total_rows",
            "value": total_rows,
            "description": "宽表总行数"
        },
        {
            "check_item": "total_columns",
            "value": total_columns,
            "description": "宽表总列数"
        },
        {
            "check_item": "total_missing_values",
            "value": total_missing_values,
            "description": "缺失值总数"
        },
        {
            "check_item": "duplicate_user_item_key_count",
            "value": duplicate_key_count,
            "description": "user_id + item_id 重复数量"
        },
        {
            "check_item": "not_buy_count",
            "value": not_buy_count,
            "description": "未购买样本数量"
        },
        {
            "check_item": "buy_count",
            "value": buy_count,
            "description": "购买样本数量"
        },
        {
            "check_item": "buy_rate",
            "value": buy_rate,
            "description": "购买样本比例"
        }
    ]

    return pd.DataFrame(summary_data)


def create_column_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成字段级检查结果。
    """
    column_results = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        unique_count = int(df[column].nunique())

        if column in ID_COLUMNS:
            column_role = "id_column"
            model_suggestion = "exclude_from_model"
            reason = "ID字段只用于连接和追踪，不作为模型特征"

        elif column == LABEL_COLUMN:
            column_role = "label"
            model_suggestion = "target_label"
            reason = "标签字段，模型预测目标"

        elif column in POSSIBLE_LEAKAGE_COLUMNS:
            column_role = "possible_leakage"
            model_suggestion = "exclude_from_model"
            reason = "疑似标签泄露字段，建模前排除"

        else:
            column_role = "feature"
            model_suggestion = "candidate_feature"
            reason = "作为候选特征，后续根据特征筛选和模型效果判断"

        column_results.append({
            "column_name": column,
            "data_type": str(df[column].dtype),
            "missing_count": missing_count,
            "unique_count": unique_count,
            "column_role": column_role,
            "model_suggestion": model_suggestion,
            "reason": reason
        })

    return pd.DataFrame(column_results)


def create_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成标签分布。
    """
    label_df = (
        df[LABEL_COLUMN]
        .value_counts()
        .reset_index()
    )

    label_df.columns = [LABEL_COLUMN, "sample_count"]
    label_df["sample_rate"] = label_df["sample_count"] / len(df)

    return label_df.sort_values(by=LABEL_COLUMN)


def create_model_feature_list(column_check_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成候选特征列表。
    """
    feature_list_df = column_check_df[
        column_check_df["model_suggestion"] == "candidate_feature"
    ][["column_name", "data_type", "model_suggestion"]]

    return feature_list_df.reset_index(drop=True)


def check_model_data() -> None:
    """
    运行建模前数据检查。
    """
    PARQUET_DIR.mkdir(exist_ok=True)

    df = read_wide_table()

    basic_summary_df = create_basic_summary(df)
    column_check_df = create_column_check(df)
    label_distribution_df = create_label_distribution(df)
    model_feature_list_df = create_model_feature_list(column_check_df)

    save_parquet_result(
        basic_summary_df,
        "05_model_data_basic_summary"
    )

    save_parquet_result(
        column_check_df,
        "05_model_data_column_check"
    )

    save_parquet_result(
        label_distribution_df,
        "05_model_data_label_distribution"
    )

    save_parquet_result(
        model_feature_list_df,
        "05_model_candidate_features"
    )

    print("建模前数据检查完成。")
    print(f"宽表行数：{df.shape[0]}")
    print(f"宽表列数：{df.shape[1]}")
    print(f"候选特征数量：{len(model_feature_list_df)}")
    print(f"缺失值总数：{int(df.isna().sum().sum())}")
    print(
        "重复 user_id + item_id 数量："
        f"{int(df.duplicated(subset=['user_id', 'item_id']).sum())}"
    )

    buy_count = int((df[LABEL_COLUMN] == 1).sum())
    not_buy_count = int((df[LABEL_COLUMN] == 0).sum())
    buy_rate = buy_count / len(df)

    print(f"is_buy = 0 数量：{not_buy_count}")
    print(f"is_buy = 1 数量：{buy_count}")
    print(f"购买样本比例：{buy_rate:.4%}")

    print("已保存：05_model_data_basic_summary.parquet")
    print("已保存：05_model_data_column_check.parquet")
    print("已保存：05_model_data_label_distribution.parquet")
    print("已保存：05_model_candidate_features.parquet")


if __name__ == "__main__":
    check_model_data()