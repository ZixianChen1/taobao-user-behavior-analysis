"""
1. 读取原始用户行为数据
2. 生成用户、商品、类目、用户商品、用户类目、时间和近期行为特征
3. 将结果保存到 output_csv 文件夹
"""

import sys
from pathlib import Path

import pandas as pd


# 将项目根目录加入 Python 搜索路径，方便读取 config.py
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))

from config import DATA_FILE, OUTPUT_DIR, TIME_FORMAT, BEHAVIOR_MAP


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    安全除法，避免分母为 0。
    """
    denominator = denominator.where(denominator != 0, pd.NA)
    return numerator.div(denominator).fillna(0)


def load_and_prepare_data() -> pd.DataFrame:
    """
    读取数据，并补充基础时间字段和行为名称。
    """
    df = pd.read_csv(DATA_FILE)

    # 转换 time 字段
    df["time"] = pd.to_datetime(df["time"], format=TIME_FORMAT, errors="coerce")

    # 删除无效时间记录
    df = df.dropna(subset=["time"]).copy()

    # 只保留合法行为类型
    df = df[df["behavior_type"].isin(BEHAVIOR_MAP.keys())].copy()

    # 映射行为名称
    df["behavior_name"] = df["behavior_type"].map(BEHAVIOR_MAP)

    # 提取日期字段
    df["date"] = df["time"].dt.date
    df["date_dt"] = df["time"].dt.normalize()

    # 提取小时字段
    df["hour"] = df["time"].dt.hour

    # 提取星期字段，0 表示周一，6 表示周日
    df["day_of_week"] = df["time"].dt.weekday

    # 判断是否周末
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 判断是否工作日
    df["is_workday"] = (1 - df["is_weekend"]).astype(int)

    # 判断是否晚间高峰时段
    df["is_peak_hour"] = df["hour"].between(20, 23).astype(int)

    # 计算距离双十二的天数
    double12_date = pd.to_datetime(df["time"].dt.year.astype(str) + "-12-12")
    df["days_to_double12"] = (double12_date - df["date_dt"]).dt.days

    # 判断是否处于双十二前后窗口，包含 12 月 10 日到 12 月 12 日
    df["is_double12_period"] = df["days_to_double12"].between(0, 2).astype(int)

    print(f"数据读取完成，数据量：{len(df)}")

    return df


def get_behavior_counts(
    df: pd.DataFrame,
    group_cols: list[str],
    prefix: str
) -> pd.DataFrame:
    """
    按指定维度统计 pv、fav、cart、buy 次数。
    """
    behavior_counts = (
        df
        .groupby(group_cols + ["behavior_name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # 保证四类行为列都存在
    for behavior in ["pv", "fav", "cart", "buy"]:
        if behavior not in behavior_counts.columns:
            behavior_counts[behavior] = 0

    # 重命名行为统计列
    behavior_counts = behavior_counts.rename(columns={
        "pv": f"{prefix}_pv_count",
        "fav": f"{prefix}_fav_count",
        "cart": f"{prefix}_cart_count",
        "buy": f"{prefix}_buy_count"
    })

    return behavior_counts


def add_total_and_rate(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    补充总互动次数、收藏率、加购率和购买率。
    """
    pv_col = f"{prefix}_pv_count"
    fav_col = f"{prefix}_fav_count"
    cart_col = f"{prefix}_cart_count"
    buy_col = f"{prefix}_buy_count"

    # 总互动次数
    df[f"{prefix}_total_interaction_count"] = (
        df[pv_col]
        + df[fav_col]
        + df[cart_col]
        + df[buy_col]
    )

    # 收藏率
    df[f"{prefix}_fav_rate"] = safe_divide(df[fav_col], df[pv_col])

    # 加购率
    df[f"{prefix}_cart_rate"] = safe_divide(df[cart_col], df[pv_col])

    # 购买率
    df[f"{prefix}_buy_rate"] = safe_divide(df[buy_col], df[pv_col])

    return df


def create_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成用户特征表，每个 user_id 一行。
    """
    user_features = get_behavior_counts(
        df=df,
        group_cols=["user_id"],
        prefix="user"
    )

    # 补充用户互动次数和转化比例
    user_features = add_total_and_rate(user_features, "user")

    # 用户活跃天数
    # 用户活跃小时数
    # 用户互动过的商品数量
    # 用户互动过的类目数量
    user_extra_features = (
        df
        .groupby("user_id")
        .agg(
            user_active_days=("date", "nunique"),
            user_active_hours=("hour", "nunique"),
            user_interacted_item_count=("item_id", "nunique"),
            user_interacted_category_count=("item_category", "nunique")
        )
        .reset_index()
    )

    # 合并用户补充特征
    user_features = user_features.merge(
        user_extra_features,
        on="user_id",
        how="left"
    )

    user_features.to_csv(
        OUTPUT_DIR / "03_user_features.csv",
        index=False
    )

    print("用户特征表已保存：03_user_features.csv")

    return user_features


def create_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成商品特征表，每个 item_id 一行。
    """
    item_features = get_behavior_counts(
        df=df,
        group_cols=["item_id"],
        prefix="item"
    )

    # 商品所属类目
    # 商品覆盖用户数
    item_extra_features = (
        df
        .groupby("item_id")
        .agg(
            item_category=("item_category", "first"),
            item_user_count=("user_id", "nunique")
        )
        .reset_index()
    )

    # 合并商品补充特征
    item_features = item_features.merge(
        item_extra_features,
        on="item_id",
        how="left"
    )

    # 补充商品互动次数和转化比例
    item_features = add_total_and_rate(item_features, "item")

    item_features.to_csv(
        OUTPUT_DIR / "03_item_features.csv",
        index=False
    )

    print("商品特征表已保存：03_item_features.csv")

    return item_features


def create_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成类目特征表，每个 item_category 一行。
    """
    category_features = get_behavior_counts(
        df=df,
        group_cols=["item_category"],
        prefix="category"
    )

    # 类目覆盖用户数
    # 类目商品数量
    category_extra_features = (
        df
        .groupby("item_category")
        .agg(
            category_user_count=("user_id", "nunique"),
            category_item_count=("item_id", "nunique")
        )
        .reset_index()
    )

    # 合并类目补充特征
    category_features = category_features.merge(
        category_extra_features,
        on="item_category",
        how="left"
    )

    # 补充类目互动次数和转化比例
    category_features = add_total_and_rate(category_features, "category")

    category_features.to_csv(
        OUTPUT_DIR / "03_category_features.csv",
        index=False
    )

    print("类目特征表已保存：03_category_features.csv")

    return category_features


def create_user_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成用户商品交互特征表，每个 user_id + item_id 一行。
    """
    user_item_features = get_behavior_counts(
        df=df,
        group_cols=["user_id", "item_id"],
        prefix="user_item"
    )

    # 用户商品对应的类目
    # 用户对该商品的最后一次行为时间
    user_item_extra_features = (
        df
        .groupby(["user_id", "item_id"])
        .agg(
            item_category=("item_category", "first"),
            last_behavior_time=("time", "max")
        )
        .reset_index()
    )

    # 提取最后一次行为小时
    user_item_extra_features["last_behavior_hour"] = (
        user_item_extra_features["last_behavior_time"].dt.hour
    )

    user_item_features = user_item_features.merge(
        user_item_extra_features[
            ["user_id", "item_id", "item_category", "last_behavior_hour"]
        ],
        on=["user_id", "item_id"],
        how="left"
    )

    # 补充用户商品互动次数和转化比例
    user_item_features = add_total_and_rate(user_item_features, "user_item")

    # 是否为晚间高峰时段
    user_item_features["is_peak_hour"] = (
        user_item_features["last_behavior_hour"]
        .between(20, 23)
        .astype(int)
    )

    # 是否购买，作为后续模型标签
    user_item_features["is_buy"] = (
        user_item_features["user_item_buy_count"] > 0
    ).astype(int)

    user_item_features.to_csv(
        OUTPUT_DIR / "03_user_item_features.csv",
        index=False
    )

    print("用户商品交互特征表已保存：03_user_item_features.csv")

    return user_item_features


def create_user_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成用户类目特征表，每个 user_id + item_category 一行。
    """
    user_category_features = get_behavior_counts(
        df=df,
        group_cols=["user_id", "item_category"],
        prefix="user_category"
    )

    # 补充用户类目互动次数和转化比例
    user_category_features = add_total_and_rate(
        user_category_features,
        "user_category"
    )

    # 用户在该类目下互动过的商品数量
    user_category_item_count = (
        df
        .groupby(["user_id", "item_category"])["item_id"]
        .nunique()
        .reset_index(name="user_category_item_count")
    )

    user_category_features = user_category_features.merge(
        user_category_item_count,
        on=["user_id", "item_category"],
        how="left"
    )

    user_category_features.to_csv(
        OUTPUT_DIR / "03_user_category_features.csv",
        index=False
    )

    print("用户类目特征表已保存：03_user_category_features.csv")

    return user_category_features


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成时间特征表，每个 date + hour 一行。
    """
    time_features = get_behavior_counts(
        df=df,
        group_cols=["date", "hour"],
        prefix="time"
    )

    # 补充时间维度互动次数和转化比例
    time_features = add_total_and_rate(time_features, "time")

    # 提取每个 date + hour 对应的时间属性
    time_flags = (
        df
        .groupby(["date", "hour"])
        .agg(
            day_of_week=("day_of_week", "first"),
            is_weekend=("is_weekend", "first"),
            is_workday=("is_workday", "first"),
            is_peak_hour=("is_peak_hour", "first"),
            days_to_double12=("days_to_double12", "first"),
            is_double12_period=("is_double12_period", "first")
        )
        .reset_index()
    )

    # 合并时间属性
    time_features = time_features.merge(
        time_flags,
        on=["date", "hour"],
        how="left"
    )

    time_features.to_csv(
        OUTPUT_DIR / "03_time_features.csv",
        index=False
    )

    print("时间特征表已保存：03_time_features.csv")

    return time_features


def count_recent_behavior(
    df: pd.DataFrame,
    group_cols: list[str],
    behavior_name: str,
    new_col_name: str
) -> pd.DataFrame:
    """
    统计指定行为在指定维度下的近期次数。
    """
    result = (
        df[df["behavior_name"] == behavior_name]
        .groupby(group_cols)
        .size()
        .reset_index(name=new_col_name)
    )

    return result


def create_recent_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成近期行为特征表，每个 user_id + item_id 一行。
    """
    max_time = df["time"].max()

    # 保留所有用户商品组合
    recent_features = (
        df[["user_id", "item_id", "item_category"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # 近 1 天数据
    recent_1d = df[df["time"] >= max_time - pd.Timedelta(days=1)].copy()

    # 近 3 天数据
    recent_3d = df[df["time"] >= max_time - pd.Timedelta(days=3)].copy()

    # 近 7 天数据
    recent_7d = df[df["time"] >= max_time - pd.Timedelta(days=7)].copy()

    # 近 1 天用户商品浏览次数
    recent_1d_user_item_pv = count_recent_behavior(
        df=recent_1d,
        group_cols=["user_id", "item_id"],
        behavior_name="pv",
        new_col_name="recent_1d_user_item_pv_count"
    )

    # 近 3 天用户商品浏览次数
    recent_3d_user_item_pv = count_recent_behavior(
        df=recent_3d,
        group_cols=["user_id", "item_id"],
        behavior_name="pv",
        new_col_name="recent_3d_user_item_pv_count"
    )

    # 近 7 天用户商品浏览次数
    recent_7d_user_item_pv = count_recent_behavior(
        df=recent_7d,
        group_cols=["user_id", "item_id"],
        behavior_name="pv",
        new_col_name="recent_7d_user_item_pv_count"
    )

    # 近 3 天用户商品加购次数
    recent_3d_user_item_cart = count_recent_behavior(
        df=recent_3d,
        group_cols=["user_id", "item_id"],
        behavior_name="cart",
        new_col_name="recent_3d_user_item_cart_count"
    )

    # 近 7 天用户类目浏览次数
    recent_7d_user_category_pv = count_recent_behavior(
        df=recent_7d,
        group_cols=["user_id", "item_category"],
        behavior_name="pv",
        new_col_name="recent_7d_user_category_pv_count"
    )

    # 近 7 天用户类目加购次数
    recent_7d_user_category_cart = count_recent_behavior(
        df=recent_7d,
        group_cols=["user_id", "item_category"],
        behavior_name="cart",
        new_col_name="recent_7d_user_category_cart_count"
    )

    # 近 7 天用户活跃天数
    recent_7d_user_active_days = (
        recent_7d
        .groupby("user_id")["date"]
        .nunique()
        .reset_index(name="recent_7d_user_active_days")
    )

    # 合并用户商品近期行为
    recent_features = recent_features.merge(
        recent_1d_user_item_pv,
        on=["user_id", "item_id"],
        how="left"
    )

    recent_features = recent_features.merge(
        recent_3d_user_item_pv,
        on=["user_id", "item_id"],
        how="left"
    )

    recent_features = recent_features.merge(
        recent_7d_user_item_pv,
        on=["user_id", "item_id"],
        how="left"
    )

    recent_features = recent_features.merge(
        recent_3d_user_item_cart,
        on=["user_id", "item_id"],
        how="left"
    )

    # 合并用户类目近期行为
    recent_features = recent_features.merge(
        recent_7d_user_category_pv,
        on=["user_id", "item_category"],
        how="left"
    )

    recent_features = recent_features.merge(
        recent_7d_user_category_cart,
        on=["user_id", "item_category"],
        how="left"
    )

    # 合并用户近期活跃天数
    recent_features = recent_features.merge(
        recent_7d_user_active_days,
        on="user_id",
        how="left"
    )

    # 近期行为缺失值填充为 0
    recent_cols = [
        col for col in recent_features.columns
        if col.startswith("recent_")
    ]

    recent_features[recent_cols] = (
        recent_features[recent_cols]
        .fillna(0)
        .astype(int)
    )

    recent_features.to_csv(
        OUTPUT_DIR / "03_recent_behavior_features.csv",
        index=False
    )

    print("近期行为特征表已保存：03_recent_behavior_features.csv")

    return recent_features


def main() -> None:
    """
    运行特征工程流程。
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_and_prepare_data()

    create_user_features(df)
    create_item_features(df)
    create_category_features(df)
    create_user_item_features(df)
    create_user_category_features(df)
    create_time_features(df)
    create_recent_behavior_features(df)

    print("第三阶段特征工程已完成。")


if __name__ == "__main__":
    main()
