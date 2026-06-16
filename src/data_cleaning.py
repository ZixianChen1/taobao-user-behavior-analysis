"""
数据清洗模块：转换时间、筛选行为类型、增加行为名称、日期和小时字段。
"""

from typing import Dict

import pandas as pd


def clean_data(
    df: pd.DataFrame,
    behavior_map: Dict[int, str],
    time_format: str
) -> pd.DataFrame:
    """
    清洗用户行为数据。
    """
    # 复制数据，避免直接修改原始数据
    df = df.copy()

    # 将 time 字段转换为 datetime 类型
    df["time"] = pd.to_datetime(
        df["time"],
        format=time_format,
        errors="coerce"
    )

    # 统计无法转换的时间数量
    invalid_time_count = df["time"].isnull().sum()
    print(f"无效时间数量：{invalid_time_count}")

    # 删除时间为空的记录
    df = df.dropna(subset=["time"])

    # 只保留合法行为类型 1、2、3、4
    df = df[df["behavior_type"].isin(behavior_map.keys())]

    # 新增行为名称字段
    df["behavior_name"] = df["behavior_type"].map(behavior_map)

    # 新增日期字段
    df["date"] = df["time"].dt.date

    # 新增小时字段
    df["hour"] = df["time"].dt.hour

    # 返回清洗后的数据
    return df