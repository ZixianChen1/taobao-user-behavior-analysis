"""
第六阶段：建模数据集准备脚本。

作用：
1. 读取 Parquet 特征宽表
2. 删除 ID 字段、标签字段和疑似数据泄露字段
3. 生成建模特征表 X 和标签表 y
4. 保存建模数据集和字段说明为 Parquet
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

WIDE_TABLE_PARQUET_FILE = "04_feature_wide_table.parquet"
WIDE_TABLE_CSV_FILE = "04_feature_wide_table.csv"

LABEL_COLUMN = "is_buy"


ID_COLUMNS = [
    "user_id",
    "item_id",
    "item_category"
]


# 为了避免数据泄露，先保守删除和 buy 直接相关的字段
POSSIBLE_LEAKAGE_COLUMNS = [
    "user_buy_count",
    "user_buy_rate",

    "item_buy_count",
    "item_buy_rate",

    "category_buy_count",
    "category_buy_rate",

    "user_item_buy_count",
    "user_item_buy_rate",

    "user_category_buy_count",
    "user_category_buy_rate",

    "time_buy_count",
    "time_buy_rate",

    # total_interaction_count 包含 buy，先保守删除
    "user_total_interaction_count",
    "item_total_interaction_count",
    "category_total_interaction_count",
    "user_item_total_interaction_count",
    "user_category_total_interaction_count"
]


def read_wide_table() -> pd.DataFrame:
    """
    读取特征宽表。
    优先读取 Parquet，如果没有则读取 CSV 并转成 Parquet。
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


def get_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    获取实际存在于宽表中的字段。
    """
    return [col for col in columns if col in df.columns]


def save_parquet(df: pd.DataFrame, file_name: str) -> None:
    """
    保存 DataFrame 为 Parquet 文件。
    """
    PARQUET_DIR.mkdir(exist_ok=True)

    df.to_parquet(
        PARQUET_DIR / file_name,
        index=False,
        engine="pyarrow"
    )


def create_excluded_columns_summary(
    id_columns: list[str],
    label_column: str,
    leakage_columns: list[str]
) -> pd.DataFrame:
    """
    生成被排除字段说明。
    """
    records = []

    for col in id_columns:
        records.append({
            "column_name": col,
            "exclude_reason": "ID字段，只用于连接和追踪，不直接用于建模"
        })

    records.append({
        "column_name": label_column,
        "exclude_reason": "标签字段，是模型预测目标，不能作为特征"
    })

    for col in leakage_columns:
        records.append({
            "column_name": col,
            "exclude_reason": "疑似数据泄露字段，和购买结果直接相关，先排除"
        })

    return pd.DataFrame(records)


def create_dataset_summary(
    original_df: pd.DataFrame,
    model_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    excluded_columns: list[str]
) -> pd.DataFrame:
    """
    生成建模数据集汇总。
    """
    buy_count = int((target_df[LABEL_COLUMN] == 1).sum())
    not_buy_count = int((target_df[LABEL_COLUMN] == 0).sum())
    buy_rate = buy_count / len(target_df) if len(target_df) > 0 else 0

    summary_data = [
        {
            "item": "original_rows",
            "value": original_df.shape[0],
            "description": "原始宽表行数"
        },
        {
            "item": "original_columns",
            "value": original_df.shape[1],
            "description": "原始宽表列数"
        },
        {
            "item": "model_dataset_rows",
            "value": model_df.shape[0],
            "description": "建模数据集行数"
        },
        {
            "item": "model_dataset_columns",
            "value": model_df.shape[1],
            "description": "建模数据集列数，包含标签"
        },
        {
            "item": "feature_count",
            "value": feature_df.shape[1],
            "description": "最终候选特征数量"
        },
        {
            "item": "excluded_column_count",
            "value": len(excluded_columns),
            "description": "被排除字段数量"
        },
        {
            "item": "missing_value_count",
            "value": int(model_df.isna().sum().sum()),
            "description": "建模数据集缺失值数量"
        },
        {
            "item": "not_buy_count",
            "value": not_buy_count,
            "description": "未购买样本数量"
        },
        {
            "item": "buy_count",
            "value": buy_count,
            "description": "购买样本数量"
        },
        {
            "item": "buy_rate",
            "value": buy_rate,
            "description": "购买样本比例"
        }
    ]

    return pd.DataFrame(summary_data)


def prepare_model_dataset() -> None:
    """
    生成建模数据集。
    """
    PARQUET_DIR.mkdir(exist_ok=True)

    wide_df = read_wide_table()

    if LABEL_COLUMN not in wide_df.columns:
        raise ValueError(f"宽表中不存在标签字段：{LABEL_COLUMN}")

    # 找出实际存在的 ID 字段和泄露字段
    existing_id_columns = get_existing_columns(wide_df, ID_COLUMNS)
    existing_leakage_columns = get_existing_columns(
        wide_df,
        POSSIBLE_LEAKAGE_COLUMNS
    )

    # 合并需要排除的字段，并去重
    excluded_columns = list(dict.fromkeys(
        existing_id_columns
        + [LABEL_COLUMN]
        + existing_leakage_columns
    ))

    # 特征 X
    feature_df = wide_df.drop(columns=excluded_columns)

    # 标签 y
    target_df = wide_df[[LABEL_COLUMN]].copy()

    # 建模数据集：特征 + 标签
    model_df = pd.concat([feature_df, target_df], axis=1)

    # 被排除字段说明
    excluded_summary_df = create_excluded_columns_summary(
        existing_id_columns,
        LABEL_COLUMN,
        existing_leakage_columns
    )

    # 建模数据集汇总
    dataset_summary_df = create_dataset_summary(
        wide_df,
        model_df,
        feature_df,
        target_df,
        excluded_columns
    )

    # 特征列表
    feature_list_df = pd.DataFrame({
        "feature_name": feature_df.columns,
        "data_type": [str(feature_df[col].dtype) for col in feature_df.columns]
    })

    # 保存 Parquet 文件
    save_parquet(model_df, "06_model_dataset.parquet")
    save_parquet(feature_df, "06_model_features.parquet")
    save_parquet(target_df, "06_model_target.parquet")
    save_parquet(excluded_summary_df, "06_excluded_columns.parquet")
    save_parquet(dataset_summary_df, "06_model_dataset_summary.parquet")
    save_parquet(feature_list_df, "06_model_feature_list.parquet")

    print("建模数据集准备完成。")
    print(f"原始宽表行数：{wide_df.shape[0]}")
    print(f"原始宽表列数：{wide_df.shape[1]}")
    print(f"最终候选特征数量：{feature_df.shape[1]}")
    print(f"被排除字段数量：{len(excluded_columns)}")
    print(f"建模数据集缺失值数量：{int(model_df.isna().sum().sum())}")

    buy_count = int((target_df[LABEL_COLUMN] == 1).sum())
    not_buy_count = int((target_df[LABEL_COLUMN] == 0).sum())
    buy_rate = buy_count / len(target_df)

    print(f"is_buy = 0 数量：{not_buy_count}")
    print(f"is_buy = 1 数量：{buy_count}")
    print(f"购买样本比例：{buy_rate:.4%}")

    print("已保存：06_model_dataset.parquet")
    print("已保存：06_model_features.parquet")
    print("已保存：06_model_target.parquet")
    print("已保存：06_excluded_columns.parquet")
    print("已保存：06_model_dataset_summary.parquet")
    print("已保存：06_model_feature_list.parquet")


if __name__ == "__main__":
    prepare_model_dataset()