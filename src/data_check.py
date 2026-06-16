"""
数据检查模块：检查前 5 行、数据类型、缺失值、唯一值和重复行。
"""

from pathlib import Path

import pandas as pd


def print_basic_info(df: pd.DataFrame) -> None:
    """
    打印数据基础信息。
    """
    print("\n===== 前 5 行数据 =====")
    # 查看前 5 行数据
    print(df.head())

    print("\n===== 数据行列数 =====")
    # 查看数据行数和列数
    print(df.shape)

    print("\n===== 字段名称 =====")
    # 查看所有字段名
    print(df.columns.tolist())

    print("\n===== 数据类型 =====")
    # 查看每一列的数据类型
    print(df.dtypes)


def save_data_types(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存每列的数据类型。
    """
    # 整理字段名和数据类型
    data_types = pd.DataFrame({
        "column_name": df.columns,
        "data_type": df.dtypes.astype(str).values
    })

    # 保存为 CSV
    data_types.to_csv(
        output_dir / "01_data_types.csv",
        index=False
    )

    print("数据类型检查结果已保存。")


def save_missing_values(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存每列缺失值数量和比例。
    """
    # 统计缺失值数量和缺失比例
    missing_values = pd.DataFrame({
        "column_name": df.columns,
        "missing_count": df.isnull().sum().values,
        "missing_ratio": df.isnull().mean().values
    })

    # 保存为 CSV
    missing_values.to_csv(
        output_dir / "01_missing_values.csv",
        index=False
    )

    print("缺失值检查结果已保存。")


def save_unique_counts(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存每列唯一值数量。
    """
    # 统计每一列有多少个不同值
    unique_counts = pd.DataFrame({
        "column_name": df.columns,
        "unique_count": df.nunique().values
    })

    # 保存为 CSV
    unique_counts.to_csv(
        output_dir / "01_unique_counts.csv",
        index=False
    )

    print("唯一值数量检查结果已保存。")


def save_duplicate_check(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存重复行检查结果。
    """
    # 统计完全重复的行数
    duplicate_count = df.duplicated().sum()

    # 计算重复行比例
    duplicate_ratio = duplicate_count / len(df)

    # 整理重复行检查结果
    duplicate_result = pd.DataFrame({
        "metric": ["duplicate_count", "duplicate_ratio"],
        "value": [duplicate_count, duplicate_ratio]
    })

    # 保存为 CSV
    duplicate_result.to_csv(
        output_dir / "01_duplicate_check.csv",
        index=False
    )

    print("重复值检查结果已保存。")