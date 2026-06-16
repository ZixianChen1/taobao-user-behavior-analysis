"""
基础统计模块：生成整体、行为、用户、商品、类目、时间和转化率统计。
"""

from pathlib import Path

import pandas as pd


def save_overall_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存整体数据概览。
    """
    # 统计总记录数、用户数、商品数、类目数、行为类型数和时间范围
    summary = pd.DataFrame({
        "metric": [
            "total_records",
            "total_users",
            "total_items",
            "total_categories",
            "total_behavior_types",
            "start_time",
            "end_time"
        ],
        "value": [
            len(df),
            df["user_id"].nunique(),
            df["item_id"].nunique(),
            df["item_category"].nunique(),
            df["behavior_type"].nunique(),
            df["time"].min(),
            df["time"].max()
        ]
    })

    # 保存为 CSV
    summary.to_csv(
        output_dir / "01_overall_summary.csv",
        index=False
    )

    print("整体数据概览已保存。")


def save_behavior_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存行为类型分布。
    """
    # 统计每种行为出现次数
    behavior_distribution = (
        df
        .groupby(["behavior_type", "behavior_name"])
        .size()
        .reset_index(name="behavior_count")
        .sort_values(by="behavior_type")
    )

    # 计算每种行为占比
    behavior_distribution["behavior_ratio"] = (
        behavior_distribution["behavior_count"] / len(df)
    )

    # 保存为 CSV
    behavior_distribution.to_csv(
        output_dir / "01_behavior_distribution.csv",
        index=False
    )

    print("行为类型分布已保存。")


def save_user_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存用户维度统计。
    """
    # 统计每个用户的总互动次数
    user_interaction_count = (
        df
        .groupby("user_id")
        .size()
        .reset_index(name="total_interaction_count")
    )

    # 统计用户互动次数的描述信息
    user_interaction_summary = (
        user_interaction_count["total_interaction_count"]
        .describe()
        .reset_index()
    )

    # 修改列名
    user_interaction_summary.columns = ["metric", "value"]

    # 统计每个用户不同类型行为次数
    user_behavior_counts = (
        df
        .groupby(["user_id", "behavior_name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # 保存每个用户总互动次数
    user_interaction_count.to_csv(
        output_dir / "01_user_interaction_count.csv",
        index=False
    )

    # 保存用户互动次数统计描述
    user_interaction_summary.to_csv(
        output_dir / "01_user_interaction_summary.csv",
        index=False
    )

    # 保存每个用户不同类型行为次数
    user_behavior_counts.to_csv(
        output_dir / "01_user_behavior_counts.csv",
        index=False
    )

    print("用户维度统计已保存。")


def save_item_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存商品维度统计。
    """
    # 统计每个商品总互动次数
    item_interaction_count = (
        df
        .groupby("item_id")
        .size()
        .reset_index(name="total_interaction_count")
    )

    # 统计商品互动次数的描述信息
    item_interaction_summary = (
        item_interaction_count["total_interaction_count"]
        .describe()
        .reset_index()
    )

    # 修改列名
    item_interaction_summary.columns = ["metric", "value"]

    # 统计每个商品不同类型行为次数
    item_behavior_counts = (
        df
        .groupby(["item_id", "behavior_name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # 保存每个商品总互动次数
    item_interaction_count.to_csv(
        output_dir / "01_item_interaction_count.csv",
        index=False
    )

    # 保存商品互动次数统计描述
    item_interaction_summary.to_csv(
        output_dir / "01_item_interaction_summary.csv",
        index=False
    )

    # 保存每个商品不同类型行为次数
    item_behavior_counts.to_csv(
        output_dir / "01_item_behavior_counts.csv",
        index=False
    )

    print("商品维度统计已保存。")


def save_category_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存类目维度统计。
    """
    # 统计每个类目的互动次数、用户数和商品数
    category_summary = (
        df
        .groupby("item_category")
        .agg(
            total_interaction_count=("user_id", "count"),
            unique_user_count=("user_id", "nunique"),
            unique_item_count=("item_id", "nunique")
        )
        .reset_index()
        .sort_values(by="total_interaction_count", ascending=False)
    )

    # 统计每个类目下不同类型行为次数
    category_behavior_counts = (
        df
        .groupby(["item_category", "behavior_name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # 保存类目整体统计
    category_summary.to_csv(
        output_dir / "01_category_summary.csv",
        index=False
    )

    # 保存类目行为统计
    category_behavior_counts.to_csv(
        output_dir / "01_category_behavior_counts.csv",
        index=False
    )

    print("类目维度统计已保存。")


def save_time_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存时间维度统计。
    """
    # 按日期和行为类型统计数量
    daily_behavior_count = (
        df
        .groupby(["date", "behavior_name"])
        .size()
        .reset_index(name="behavior_count")
        .sort_values(by=["date", "behavior_name"])
    )

    # 按小时和行为类型统计数量
    hourly_behavior_count = (
        df
        .groupby(["hour", "behavior_name"])
        .size()
        .reset_index(name="behavior_count")
        .sort_values(by=["hour", "behavior_name"])
    )

    # 保存每日行为统计
    daily_behavior_count.to_csv(
        output_dir / "01_daily_behavior_count.csv",
        index=False
    )

    # 保存每小时行为统计
    hourly_behavior_count.to_csv(
        output_dir / "01_hourly_behavior_count.csv",
        index=False
    )

    print("时间维度统计已保存。")


def save_conversion_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    保存简单转化率统计。
    """
    # 统计各行为出现次数
    behavior_counts = df["behavior_name"].value_counts()

    # 获取浏览次数
    pv_count = behavior_counts.get("pv", 0)

    # 获取收藏次数
    fav_count = behavior_counts.get("fav", 0)

    # 获取加购次数
    cart_count = behavior_counts.get("cart", 0)

    # 获取购买次数
    buy_count = behavior_counts.get("buy", 0)

    # 计算简单转化指标
    conversion_summary = pd.DataFrame({
        "metric": [
            "pv_count",
            "fav_count",
            "cart_count",
            "buy_count",
            "buy_pv_ratio",
            "buy_cart_ratio",
            "buy_fav_ratio"
        ],
        "value": [
            pv_count,
            fav_count,
            cart_count,
            buy_count,
            buy_count / pv_count if pv_count > 0 else 0,
            buy_count / cart_count if cart_count > 0 else 0,
            buy_count / fav_count if fav_count > 0 else 0
        ]
    })

    # 保存为 CSV
    conversion_summary.to_csv(
        output_dir / "01_conversion_summary.csv",
        index=False
    )

    print("转化率统计已保存。")