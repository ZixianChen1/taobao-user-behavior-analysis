"""
数据检查模块：检查前 5 行、数据类型、缺失值、唯一值、重复行和数据合理性。
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


def save_validity_check(
    df: pd.DataFrame,
    output_dir: Path,
    valid_behavior_types: list[int],
    time_format: str
) -> None:
    """
    保存数据合理性检查结果。

    主要检查：
    1. behavior_type 是否只包含合法行为类型 1、2、3、4
    2. time 是否可以正常转换成时间格式
    3. time 是否存在错误日期，例如 2025-12-32
    4. time 是否存在错误小时，例如 25 点

    参数
    ----------
    df : pd.DataFrame
        原始用户行为数据。
    output_dir : Path
        输出结果文件夹路径。
    valid_behavior_types : list[int]
        合法行为类型列表，例如 [1, 2, 3, 4]。
    time_format : str
        time 字段的时间格式，例如 "%Y-%m-%d %H"。
    """
    # 检查 behavior_type 是否只包含 1、2、3、4
    invalid_behavior_mask = ~df["behavior_type"].isin(valid_behavior_types)

    # 统计非法行为类型的记录数量
    invalid_behavior_count = invalid_behavior_mask.sum()

    # 提取非法行为类型的具体取值
    invalid_behavior_values = (
        df.loc[invalid_behavior_mask, "behavior_type"]
        .dropna()
        .unique()
        .tolist()
    )

    # 将 time 字段尝试转换为 datetime
    # 如果出现错误日期或错误小时，会被转换为 NaT
    converted_time = pd.to_datetime(
        df["time"],
        format=time_format,
        errors="coerce"
    )

    # 统计无法转换成功的时间数量
    invalid_time_count = converted_time.isnull().sum()

    # 提取前 10 个无效时间样例，方便查看问题
    invalid_time_examples = (
        df.loc[converted_time.isnull(), "time"]
        .dropna()
        .head(10)
        .tolist()
    )

    # 提取转换成功的时间
    valid_time = converted_time.dropna()

    # 检查小时是否在 0 到 23 之间
    # 正常一天的小时范围是 0-23，不是 0-24
    invalid_hour_count = (
        ~valid_time.dt.hour.between(0, 23)
    ).sum()

    # 检查日期是否成功解析
    # 例如 2025-12-32、2025-13-01 都会在 invalid_time_count 中体现
    invalid_date_count = invalid_time_count

    # 整理合理性检查结果
    validity_result = pd.DataFrame({
        "check_item": [
            "invalid_behavior_type_count",
            "invalid_behavior_type_values",
            "invalid_time_count",
            "invalid_time_examples",
            "invalid_hour_count",
            "invalid_date_count"
        ],
        "result": [
            invalid_behavior_count,
            str(invalid_behavior_values),
            invalid_time_count,
            str(invalid_time_examples),
            invalid_hour_count,
            invalid_date_count
        ]
    })

    # 保存为 CSV
    validity_result.to_csv(
        output_dir / "01_validity_check.csv",
        index=False
    )

    print("数据合理性检查结果已保存。")