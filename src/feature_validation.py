"""
1. 检查 7 张特征中间表是否成功生成
2. 检查每张特征表的 key 是否唯一
3. 检查缺失值、重复行和字段完整性
4. 检查标签字段和二分类字段是否合理
5. 检查数值型特征是否存在负数
6. 抽样回查原始数据，验证部分特征计算结果
7. 输出每张特征表的 key value 说明
"""

import sys
from pathlib import Path

import pandas as pd


# 获取项目根目录路径
BASE_PATH = Path(__file__).resolve().parent.parent

# 将项目根目录加入 Python 搜索路径
sys.path.append(str(BASE_PATH))

# 读取项目配置
from config import DATA_FILE, OUTPUT_DIR, TIME_FORMAT, BEHAVIOR_MAP


# 抽样校验数量
SAMPLE_SIZE = 5


def add_validation_result(
    results: list[dict],
    check_item: str,
    expected,
    actual,
    is_passed: bool
) -> None:
    """
    添加一条汇总校验结果。
    """
    results.append({
        "check_item": check_item,
        "expected": expected,
        "actual": actual,
        "is_passed": bool(is_passed)
    })


def add_sample_result(
    results: list[dict],
    table_name: str,
    key_value: str,
    feature_name: str,
    expected_from_source_data,
    actual_from_feature_table,
    is_passed: bool
) -> None:
    """
    添加一条抽样校验结果。

    expected_from_source_data：从原始数据重新计算得到的结果。
    actual_from_feature_table：特征表中已经生成的结果。
    is_passed：判断两个结果是否一致。
    """
    results.append({
        "table_name": table_name,
        "key_value": key_value,
        "feature_name": feature_name,
        "expected_from_source_data": expected_from_source_data,
        "actual_from_feature_table": actual_from_feature_table,
        "is_passed": bool(is_passed)
    })


def read_csv_file(file_path: Path) -> pd.DataFrame:
    """
    读取 CSV 文件。
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"文件不存在：{file_path}")
        raise
    except Exception as error:
        print(f"读取文件出错：{file_path}")
        print(error)
        raise


def load_source_data() -> pd.DataFrame:
    """
    读取原始数据，并补充校验需要的基础字段。
    """
    # 读取原始数据
    df = read_csv_file(DATA_FILE)

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

    # 提取日期 datetime 字段
    df["date_dt"] = df["time"].dt.normalize()

    # 提取小时字段
    df["hour"] = df["time"].dt.hour

    # 提取星期字段
    df["day_of_week"] = df["time"].dt.weekday

    # 判断是否周末
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 判断是否工作日
    df["is_workday"] = (1 - df["is_weekend"]).astype(int)

    # 判断是否晚间高峰时段
    df["is_peak_hour"] = df["hour"].between(20, 23).astype(int)

    # 计算距离双十二天数
    double12_date = pd.to_datetime(df["time"].dt.year.astype(str) + "-12-12")
    df["days_to_double12"] = (double12_date - df["date_dt"]).dt.days

    # 判断是否双十二窗口
    df["is_double12_period"] = df["days_to_double12"].between(0, 2).astype(int)

    return df


def get_feature_file_config() -> dict:
    """
    设置特征文件、key 字段和必需字段。
    """
    return {
        "03_user_features.csv": {
            "key_cols": ["user_id"],
            "required_cols": [
                "user_id",
                "user_pv_count",
                "user_fav_count",
                "user_cart_count",
                "user_buy_count",
                "user_total_interaction_count",
                "user_fav_rate",
                "user_cart_rate",
                "user_buy_rate",
                "user_active_days",
                "user_active_hours",
                "user_interacted_item_count",
                "user_interacted_category_count"
            ]
        },
        "03_item_features.csv": {
            "key_cols": ["item_id"],
            "required_cols": [
                "item_id",
                "item_category",
                "item_pv_count",
                "item_fav_count",
                "item_cart_count",
                "item_buy_count",
                "item_total_interaction_count",
                "item_fav_rate",
                "item_cart_rate",
                "item_buy_rate",
                "item_user_count"
            ]
        },
        "03_category_features.csv": {
            "key_cols": ["item_category"],
            "required_cols": [
                "item_category",
                "category_pv_count",
                "category_fav_count",
                "category_cart_count",
                "category_buy_count",
                "category_total_interaction_count",
                "category_fav_rate",
                "category_cart_rate",
                "category_buy_rate",
                "category_user_count",
                "category_item_count"
            ]
        },
        "03_user_item_features.csv": {
            "key_cols": ["user_id", "item_id"],
            "required_cols": [
                "user_id",
                "item_id",
                "item_category",
                "user_item_pv_count",
                "user_item_fav_count",
                "user_item_cart_count",
                "user_item_buy_count",
                "user_item_total_interaction_count",
                "user_item_fav_rate",
                "user_item_cart_rate",
                "user_item_buy_rate",
                "last_behavior_hour",
                "is_peak_hour",
                "is_buy"
            ]
        },
        "03_user_category_features.csv": {
            "key_cols": ["user_id", "item_category"],
            "required_cols": [
                "user_id",
                "item_category",
                "user_category_pv_count",
                "user_category_fav_count",
                "user_category_cart_count",
                "user_category_buy_count",
                "user_category_total_interaction_count",
                "user_category_fav_rate",
                "user_category_cart_rate",
                "user_category_buy_rate",
                "user_category_item_count"
            ]
        },
        "03_time_features.csv": {
            "key_cols": ["date", "hour"],
            "required_cols": [
                "date",
                "hour",
                "time_pv_count",
                "time_fav_count",
                "time_cart_count",
                "time_buy_count",
                "time_total_interaction_count",
                "time_fav_rate",
                "time_cart_rate",
                "time_buy_rate",
                "day_of_week",
                "is_weekend",
                "is_workday",
                "is_peak_hour",
                "days_to_double12",
                "is_double12_period"
            ]
        },
        "03_recent_behavior_features.csv": {
            "key_cols": ["user_id", "item_id"],
            "required_cols": [
                "user_id",
                "item_id",
                "item_category",
                "recent_1d_user_item_pv_count",
                "recent_3d_user_item_pv_count",
                "recent_7d_user_item_pv_count",
                "recent_3d_user_item_cart_count",
                "recent_7d_user_category_pv_count",
                "recent_7d_user_category_cart_count",
                "recent_7d_user_active_days"
            ]
        }
    }


def read_feature_tables(
    feature_config: dict,
    validation_results: list[dict]
) -> dict:
    """
    读取所有特征中间表。
    """
    loaded_tables = {}

    for file_name in feature_config:
        # 设置文件路径
        file_path = OUTPUT_DIR / file_name

        # 检查文件是否存在
        file_exists = file_path.exists()

        # 记录文件存在检查结果
        add_validation_result(
            validation_results,
            f"{file_name}_exists",
            "file exists",
            "file exists" if file_exists else "file missing",
            file_exists
        )

        # 文件存在时读取特征表
        if file_exists:
            loaded_tables[file_name] = read_csv_file(file_path)

    return loaded_tables


def check_required_columns(
    table_name: str,
    df: pd.DataFrame,
    required_cols: list[str],
    validation_results: list[dict]
) -> None:
    """
    检查特征表必需字段是否完整。
    """
    # 获取缺失字段
    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    # 记录字段完整性检查结果
    add_validation_result(
        validation_results,
        f"{table_name}_required_columns",
        "no missing columns",
        str(missing_cols),
        len(missing_cols) == 0
    )


def check_unique_key(
    table_name: str,
    df: pd.DataFrame,
    key_cols: list[str],
    validation_results: list[dict]
) -> None:
    """
    检查特征表 key 是否唯一。
    """
    # 统计重复 key 数量
    duplicate_key_count = df.duplicated(subset=key_cols).sum()

    # 记录 key 唯一性检查结果
    add_validation_result(
        validation_results,
        f"{table_name}_unique_key",
        0,
        int(duplicate_key_count),
        duplicate_key_count == 0
    )


def check_missing_values(
    table_name: str,
    df: pd.DataFrame,
    validation_results: list[dict]
) -> None:
    """
    检查特征表缺失值数量。
    """
    # 统计全部缺失值数量
    missing_count = df.isnull().sum().sum()

    # 记录缺失值检查结果
    add_validation_result(
        validation_results,
        f"{table_name}_missing_values",
        0,
        int(missing_count),
        missing_count == 0
    )


def check_duplicate_rows(
    table_name: str,
    df: pd.DataFrame,
    validation_results: list[dict]
) -> None:
    """
    检查特征表完全重复行数量。
    """
    # 统计完全重复行数量
    duplicate_row_count = df.duplicated().sum()

    # 记录重复行检查结果
    add_validation_result(
        validation_results,
        f"{table_name}_duplicate_rows",
        0,
        int(duplicate_row_count),
        duplicate_row_count == 0
    )


def check_row_count(
    table_name: str,
    df: pd.DataFrame,
    key_cols: list[str],
    source_df: pd.DataFrame,
    validation_results: list[dict]
) -> None:
    """
    检查特征表行数是否等于原始数据中的唯一 key 数量。
    """
    # 统计原始数据中的唯一 key 数量
    expected_count = source_df[key_cols].drop_duplicates().shape[0]

    # 统计特征表行数
    actual_count = len(df)

    # 记录行数检查结果
    add_validation_result(
        validation_results,
        f"{table_name}_row_count",
        int(expected_count),
        int(actual_count),
        expected_count == actual_count
    )


def check_binary_columns(
    table_name: str,
    df: pd.DataFrame,
    binary_cols: list[str],
    validation_results: list[dict]
) -> None:
    """
    检查二分类字段是否只包含 0 和 1。
    """
    for col in binary_cols:
        # 字段不存在时跳过
        if col not in df.columns:
            continue

        # 获取字段实际取值
        actual_values = set(df[col].dropna().unique().tolist())

        # 判断实际取值是否只包含 0 和 1
        is_passed = actual_values.issubset({0, 1})

        # 记录二分类字段检查结果
        add_validation_result(
            validation_results,
            f"{table_name}_{col}_binary",
            "0 or 1",
            str(sorted(actual_values)),
            is_passed
        )


def check_non_negative_columns(
    table_name: str,
    df: pd.DataFrame,
    validation_results: list[dict]
) -> None:
    """
    检查数值型特征是否存在负数。
    """
    # 选择数值型字段
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # 排除允许出现负数的字段
    exclude_cols = ["days_to_double12"]

    # 确定需要检查的数值字段
    check_cols = [
        col for col in numeric_cols
        if col not in exclude_cols
    ]

    # 统计负数数量
    negative_count = (df[check_cols] < 0).sum().sum()

    # 记录非负检查结果
    add_validation_result(
        validation_results,
        f"{table_name}_non_negative_numeric_features",
        0,
        int(negative_count),
        negative_count == 0
    )


def check_rate_columns(
    table_name: str,
    df: pd.DataFrame,
    validation_results: list[dict]
) -> None:
    """
    检查比例字段是否存在负数。
    """
    # 选择比例字段
    rate_cols = [
        col for col in df.columns
        if col.endswith("_rate")
    ]

    # 无比例字段时跳过
    if len(rate_cols) == 0:
        return

    # 统计负比例数量
    negative_rate_count = (df[rate_cols] < 0).sum().sum()

    # 记录比例字段检查结果
    add_validation_result(
        validation_results,
        f"{table_name}_rate_columns_non_negative",
        0,
        int(negative_rate_count),
        negative_rate_count == 0
    )


def create_key_summary(
    feature_config: dict,
    loaded_tables: dict
) -> pd.DataFrame:
    """
    生成每张特征表的 key value 说明。
    """
    key_summary = []

    for table_name, config in feature_config.items():
        # 文件不存在时跳过
        if table_name not in loaded_tables:
            continue

        # 获取 key 字段
        key_cols = config["key_cols"]

        # 读取当前特征表
        df = loaded_tables[table_name]

        # 添加当前特征表 key 说明
        key_summary.append({
            "table_name": table_name,
            "key_columns": " + ".join(key_cols),
            "row_count": len(df),
            "column_count": df.shape[1],
            "is_key_unique": not df.duplicated(subset=key_cols).any(),
            "usage": "用于后续特征宽表连接"
        })

    return pd.DataFrame(key_summary)


def get_behavior_counts(sub_df: pd.DataFrame) -> dict:
    """
    统计一个样本范围内的 pv、fav、cart、buy 次数。
    """
    # 统计行为名称出现次数
    counts = sub_df["behavior_name"].value_counts()

    # 返回四类行为数量
    return {
        "pv": int(counts.get("pv", 0)),
        "fav": int(counts.get("fav", 0)),
        "cart": int(counts.get("cart", 0)),
        "buy": int(counts.get("buy", 0))
    }


def sample_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    从特征表中抽取少量样本行。
    """
    # 数据行数不超过抽样数量时返回全部数据
    if len(df) <= SAMPLE_SIZE:
        return df.copy()

    # 固定随机种子抽样，保证每次结果一致
    return df.sample(n=SAMPLE_SIZE, random_state=42).copy()


def validate_user_sample(
    feature_df: pd.DataFrame,
    source_df: pd.DataFrame,
    sample_results: list[dict]
) -> None:
    """
    抽样校验用户特征。
    """
    for _, row in sample_rows(feature_df).iterrows():
        # 获取样本 user_id
        user_id = row["user_id"]

        # 从原始数据筛选该用户的所有行为
        sub_df = source_df[source_df["user_id"] == user_id]

        # 从原始数据重新统计该用户的行为次数
        counts = get_behavior_counts(sub_df)

        # 设置样本 key value
        key_value = f"user_id={user_id}"

        # 校验用户浏览次数是否一致
        add_sample_result(
            sample_results,
            "03_user_features.csv",
            key_value,
            "user_pv_count",
            counts["pv"],
            int(row["user_pv_count"]),
            counts["pv"] == int(row["user_pv_count"])
        )

        # 校验用户加购次数是否一致
        add_sample_result(
            sample_results,
            "03_user_features.csv",
            key_value,
            "user_cart_count",
            counts["cart"],
            int(row["user_cart_count"]),
            counts["cart"] == int(row["user_cart_count"])
        )

        # 校验用户购买次数是否一致
        add_sample_result(
            sample_results,
            "03_user_features.csv",
            key_value,
            "user_buy_count",
            counts["buy"],
            int(row["user_buy_count"]),
            counts["buy"] == int(row["user_buy_count"])
        )

        # 从原始数据重新统计该用户活跃天数
        expected_active_days = sub_df["date"].nunique()

        # 校验用户活跃天数是否一致
        add_sample_result(
            sample_results,
            "03_user_features.csv",
            key_value,
            "user_active_days",
            int(expected_active_days),
            int(row["user_active_days"]),
            expected_active_days == int(row["user_active_days"])
        )


def validate_item_sample(
    feature_df: pd.DataFrame,
    source_df: pd.DataFrame,
    sample_results: list[dict]
) -> None:
    """
    抽样校验商品特征。
    """
    for _, row in sample_rows(feature_df).iterrows():
        # 获取样本 item_id
        item_id = row["item_id"]

        # 从原始数据筛选该商品的所有行为
        sub_df = source_df[source_df["item_id"] == item_id]

        # 从原始数据重新统计该商品的行为次数
        counts = get_behavior_counts(sub_df)

        # 设置样本 key value
        key_value = f"item_id={item_id}"

        # 校验商品浏览次数是否一致
        add_sample_result(
            sample_results,
            "03_item_features.csv",
            key_value,
            "item_pv_count",
            counts["pv"],
            int(row["item_pv_count"]),
            counts["pv"] == int(row["item_pv_count"])
        )

        # 校验商品购买次数是否一致
        add_sample_result(
            sample_results,
            "03_item_features.csv",
            key_value,
            "item_buy_count",
            counts["buy"],
            int(row["item_buy_count"]),
            counts["buy"] == int(row["item_buy_count"])
        )

        # 从原始数据重新统计该商品覆盖用户数
        expected_user_count = sub_df["user_id"].nunique()

        # 校验商品覆盖用户数是否一致
        add_sample_result(
            sample_results,
            "03_item_features.csv",
            key_value,
            "item_user_count",
            int(expected_user_count),
            int(row["item_user_count"]),
            expected_user_count == int(row["item_user_count"])
        )


def validate_category_sample(
    feature_df: pd.DataFrame,
    source_df: pd.DataFrame,
    sample_results: list[dict]
) -> None:
    """
    抽样校验类目特征。
    """
    for _, row in sample_rows(feature_df).iterrows():
        # 获取样本 item_category
        item_category = row["item_category"]

        # 从原始数据筛选该类目的所有行为
        sub_df = source_df[source_df["item_category"] == item_category]

        # 从原始数据重新统计该类目的行为次数
        counts = get_behavior_counts(sub_df)

        # 设置样本 key value
        key_value = f"item_category={item_category}"

        # 校验类目浏览次数是否一致
        add_sample_result(
            sample_results,
            "03_category_features.csv",
            key_value,
            "category_pv_count",
            counts["pv"],
            int(row["category_pv_count"]),
            counts["pv"] == int(row["category_pv_count"])
        )

        # 校验类目购买次数是否一致
        add_sample_result(
            sample_results,
            "03_category_features.csv",
            key_value,
            "category_buy_count",
            counts["buy"],
            int(row["category_buy_count"]),
            counts["buy"] == int(row["category_buy_count"])
        )

        # 从原始数据重新统计该类目商品数量
        expected_item_count = sub_df["item_id"].nunique()

        # 校验类目商品数量是否一致
        add_sample_result(
            sample_results,
            "03_category_features.csv",
            key_value,
            "category_item_count",
            int(expected_item_count),
            int(row["category_item_count"]),
            expected_item_count == int(row["category_item_count"])
        )


def validate_user_item_sample(
    feature_df: pd.DataFrame,
    source_df: pd.DataFrame,
    sample_results: list[dict]
) -> None:
    """
    抽样校验用户商品特征。
    """
    for _, row in sample_rows(feature_df).iterrows():
        # 获取样本 user_id 和 item_id
        user_id = row["user_id"]
        item_id = row["item_id"]

        # 从原始数据筛选该用户对该商品的所有行为
        sub_df = source_df[
            (source_df["user_id"] == user_id)
            & (source_df["item_id"] == item_id)
        ]

        # 从原始数据重新统计该用户商品组合的行为次数
        counts = get_behavior_counts(sub_df)

        # 设置样本 key value
        key_value = f"user_id={user_id}, item_id={item_id}"

        # 校验用户商品浏览次数是否一致
        add_sample_result(
            sample_results,
            "03_user_item_features.csv",
            key_value,
            "user_item_pv_count",
            counts["pv"],
            int(row["user_item_pv_count"]),
            counts["pv"] == int(row["user_item_pv_count"])
        )

        # 校验用户商品加购次数是否一致
        add_sample_result(
            sample_results,
            "03_user_item_features.csv",
            key_value,
            "user_item_cart_count",
            counts["cart"],
            int(row["user_item_cart_count"]),
            counts["cart"] == int(row["user_item_cart_count"])
        )

        # 从原始数据重新判断该用户商品组合是否购买
        expected_is_buy = int(counts["buy"] > 0)

        # 校验 is_buy 标签是否一致
        add_sample_result(
            sample_results,
            "03_user_item_features.csv",
            key_value,
            "is_buy",
            expected_is_buy,
            int(row["is_buy"]),
            expected_is_buy == int(row["is_buy"])
        )


def validate_user_category_sample(
    feature_df: pd.DataFrame,
    source_df: pd.DataFrame,
    sample_results: list[dict]
) -> None:
    """
    抽样校验用户类目特征。
    """
    for _, row in sample_rows(feature_df).iterrows():
        # 获取样本 user_id 和 item_category
        user_id = row["user_id"]
        item_category = row["item_category"]

        # 从原始数据筛选该用户在该类目下的所有行为
        sub_df = source_df[
            (source_df["user_id"] == user_id)
            & (source_df["item_category"] == item_category)
        ]

        # 从原始数据重新统计该用户类目组合的行为次数
        counts = get_behavior_counts(sub_df)

        # 设置样本 key value
        key_value = f"user_id={user_id}, item_category={item_category}"

        # 校验用户类目浏览次数是否一致
        add_sample_result(
            sample_results,
            "03_user_category_features.csv",
            key_value,
            "user_category_pv_count",
            counts["pv"],
            int(row["user_category_pv_count"]),
            counts["pv"] == int(row["user_category_pv_count"])
        )

        # 校验用户类目加购次数是否一致
        add_sample_result(
            sample_results,
            "03_user_category_features.csv",
            key_value,
            "user_category_cart_count",
            counts["cart"],
            int(row["user_category_cart_count"]),
            counts["cart"] == int(row["user_category_cart_count"])
        )

        # 从原始数据重新统计该用户类目下互动过的商品数量
        expected_item_count = sub_df["item_id"].nunique()

        # 校验用户类目商品数量是否一致
        add_sample_result(
            sample_results,
            "03_user_category_features.csv",
            key_value,
            "user_category_item_count",
            int(expected_item_count),
            int(row["user_category_item_count"]),
            expected_item_count == int(row["user_category_item_count"])
        )


def validate_time_sample(
    feature_df: pd.DataFrame,
    source_df: pd.DataFrame,
    sample_results: list[dict]
) -> None:
    """
    抽样校验时间特征。
    """
    for _, row in sample_rows(feature_df).iterrows():
        # 获取样本 date 和 hour
        date_value = row["date"]
        hour_value = row["hour"]

        # 从原始数据筛选该日期和小时的所有行为
        sub_df = source_df[
            (source_df["date"].astype(str) == str(date_value))
            & (source_df["hour"] == hour_value)
        ]

        # 从原始数据重新统计该日期小时的行为次数
        counts = get_behavior_counts(sub_df)

        # 设置样本 key value
        key_value = f"date={date_value}, hour={hour_value}"

        # 校验该日期小时浏览次数是否一致
        add_sample_result(
            sample_results,
            "03_time_features.csv",
            key_value,
            "time_pv_count",
            counts["pv"],
            int(row["time_pv_count"]),
            counts["pv"] == int(row["time_pv_count"])
        )

        # 校验该日期小时购买次数是否一致
        add_sample_result(
            sample_results,
            "03_time_features.csv",
            key_value,
            "time_buy_count",
            counts["buy"],
            int(row["time_buy_count"]),
            counts["buy"] == int(row["time_buy_count"])
        )

        # 从 hour 字段重新判断是否晚间高峰
        expected_peak = int(int(hour_value) in [20, 21, 22, 23])

        # 校验高峰时段标记是否一致
        add_sample_result(
            sample_results,
            "03_time_features.csv",
            key_value,
            "is_peak_hour",
            expected_peak,
            int(row["is_peak_hour"]),
            expected_peak == int(row["is_peak_hour"])
        )


def validate_recent_sample(
    feature_df: pd.DataFrame,
    source_df: pd.DataFrame,
    sample_results: list[dict]
) -> None:
    """
    抽样校验近期行为特征。
    """
    # 获取原始数据最大时间
    max_time = source_df["time"].max()

    # 筛选近 1 天原始数据
    recent_1d = source_df[source_df["time"] >= max_time - pd.Timedelta(days=1)]

    # 筛选近 3 天原始数据
    recent_3d = source_df[source_df["time"] >= max_time - pd.Timedelta(days=3)]

    # 筛选近 7 天原始数据
    recent_7d = source_df[source_df["time"] >= max_time - pd.Timedelta(days=7)]

    for _, row in sample_rows(feature_df).iterrows():
        # 获取样本 key
        user_id = row["user_id"]
        item_id = row["item_id"]
        item_category = row["item_category"]

        # 设置样本 key value
        key_value = f"user_id={user_id}, item_id={item_id}"

        # 筛选近 1 天该用户商品行为
        sub_1d_item = recent_1d[
            (recent_1d["user_id"] == user_id)
            & (recent_1d["item_id"] == item_id)
        ]

        # 筛选近 3 天该用户商品行为
        sub_3d_item = recent_3d[
            (recent_3d["user_id"] == user_id)
            & (recent_3d["item_id"] == item_id)
        ]

        # 筛选近 7 天该用户商品行为
        sub_7d_item = recent_7d[
            (recent_7d["user_id"] == user_id)
            & (recent_7d["item_id"] == item_id)
        ]

        # 筛选近 7 天该用户类目行为
        sub_7d_category = recent_7d[
            (recent_7d["user_id"] == user_id)
            & (recent_7d["item_category"] == item_category)
        ]

        # 筛选近 7 天该用户全部行为
        sub_7d_user = recent_7d[recent_7d["user_id"] == user_id]

        # 从原始数据重新统计近 1 天用户商品浏览次数
        expected_1d_pv = int((sub_1d_item["behavior_name"] == "pv").sum())

        # 校验近 1 天用户商品浏览次数是否一致
        add_sample_result(
            sample_results,
            "03_recent_behavior_features.csv",
            key_value,
            "recent_1d_user_item_pv_count",
            expected_1d_pv,
            int(row["recent_1d_user_item_pv_count"]),
            expected_1d_pv == int(row["recent_1d_user_item_pv_count"])
        )

        # 从原始数据重新统计近 3 天用户商品浏览次数
        expected_3d_pv = int((sub_3d_item["behavior_name"] == "pv").sum())

        # 校验近 3 天用户商品浏览次数是否一致
        add_sample_result(
            sample_results,
            "03_recent_behavior_features.csv",
            key_value,
            "recent_3d_user_item_pv_count",
            expected_3d_pv,
            int(row["recent_3d_user_item_pv_count"]),
            expected_3d_pv == int(row["recent_3d_user_item_pv_count"])
        )

        # 从原始数据重新统计近 7 天用户商品浏览次数
        expected_7d_pv = int((sub_7d_item["behavior_name"] == "pv").sum())

        # 校验近 7 天用户商品浏览次数是否一致
        add_sample_result(
            sample_results,
            "03_recent_behavior_features.csv",
            key_value,
            "recent_7d_user_item_pv_count",
            expected_7d_pv,
            int(row["recent_7d_user_item_pv_count"]),
            expected_7d_pv == int(row["recent_7d_user_item_pv_count"])
        )

        # 从原始数据重新统计近 3 天用户商品加购次数
        expected_3d_cart = int((sub_3d_item["behavior_name"] == "cart").sum())

        # 校验近 3 天用户商品加购次数是否一致
        add_sample_result(
            sample_results,
            "03_recent_behavior_features.csv",
            key_value,
            "recent_3d_user_item_cart_count",
            expected_3d_cart,
            int(row["recent_3d_user_item_cart_count"]),
            expected_3d_cart == int(row["recent_3d_user_item_cart_count"])
        )

        # 从原始数据重新统计近 7 天用户类目浏览次数
        expected_7d_category_pv = int(
            (sub_7d_category["behavior_name"] == "pv").sum()
        )

        # 校验近 7 天用户类目浏览次数是否一致
        add_sample_result(
            sample_results,
            "03_recent_behavior_features.csv",
            key_value,
            "recent_7d_user_category_pv_count",
            expected_7d_category_pv,
            int(row["recent_7d_user_category_pv_count"]),
            expected_7d_category_pv == int(row["recent_7d_user_category_pv_count"])
        )

        # 从原始数据重新统计近 7 天用户类目加购次数
        expected_7d_category_cart = int(
            (sub_7d_category["behavior_name"] == "cart").sum()
        )

        # 校验近 7 天用户类目加购次数是否一致
        add_sample_result(
            sample_results,
            "03_recent_behavior_features.csv",
            key_value,
            "recent_7d_user_category_cart_count",
            expected_7d_category_cart,
            int(row["recent_7d_user_category_cart_count"]),
            expected_7d_category_cart
            == int(row["recent_7d_user_category_cart_count"])
        )

        # 从原始数据重新统计近 7 天用户活跃天数
        expected_active_days = sub_7d_user["date"].nunique()

        # 校验近 7 天用户活跃天数是否一致
        add_sample_result(
            sample_results,
            "03_recent_behavior_features.csv",
            key_value,
            "recent_7d_user_active_days",
            int(expected_active_days),
            int(row["recent_7d_user_active_days"]),
            expected_active_days == int(row["recent_7d_user_active_days"])
        )


def run_sample_validation(
    loaded_tables: dict,
    source_df: pd.DataFrame
) -> pd.DataFrame:
    """
    运行所有特征表的抽样校验。
    """
    sample_results = []

    # 抽样校验用户特征表
    if "03_user_features.csv" in loaded_tables:
        validate_user_sample(
            loaded_tables["03_user_features.csv"],
            source_df,
            sample_results
        )

    # 抽样校验商品特征表
    if "03_item_features.csv" in loaded_tables:
        validate_item_sample(
            loaded_tables["03_item_features.csv"],
            source_df,
            sample_results
        )

    # 抽样校验类目特征表
    if "03_category_features.csv" in loaded_tables:
        validate_category_sample(
            loaded_tables["03_category_features.csv"],
            source_df,
            sample_results
        )

    # 抽样校验用户商品特征表
    if "03_user_item_features.csv" in loaded_tables:
        validate_user_item_sample(
            loaded_tables["03_user_item_features.csv"],
            source_df,
            sample_results
        )

    # 抽样校验用户类目特征表
    if "03_user_category_features.csv" in loaded_tables:
        validate_user_category_sample(
            loaded_tables["03_user_category_features.csv"],
            source_df,
            sample_results
        )

    # 抽样校验时间特征表
    if "03_time_features.csv" in loaded_tables:
        validate_time_sample(
            loaded_tables["03_time_features.csv"],
            source_df,
            sample_results
        )

    # 抽样校验近期行为特征表
    if "03_recent_behavior_features.csv" in loaded_tables:
        validate_recent_sample(
            loaded_tables["03_recent_behavior_features.csv"],
            source_df,
            sample_results
        )

    return pd.DataFrame(sample_results)


def validate_feature_outputs() -> None:
    """
    运行第三阶段特征工程校验流程。
    """
    # 创建输出文件夹
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 读取并处理原始数据
    source_df = load_source_data()

    # 获取特征文件配置
    feature_config = get_feature_file_config()

    # 存放汇总校验结果
    validation_results = []

    # 读取特征中间表
    loaded_tables = read_feature_tables(
        feature_config,
        validation_results
    )

    # 逐张检查特征表
    for table_name, config in feature_config.items():
        # 文件不存在时跳过
        if table_name not in loaded_tables:
            continue

        # 读取当前特征表
        df = loaded_tables[table_name]

        # 获取当前表的 key 字段
        key_cols = config["key_cols"]

        # 获取当前表的必需字段
        required_cols = config["required_cols"]

        # 检查必需字段是否完整
        check_required_columns(
            table_name,
            df,
            required_cols,
            validation_results
        )

        # 检查 key 是否唯一
        check_unique_key(
            table_name,
            df,
            key_cols,
            validation_results
        )

        # 检查缺失值数量
        check_missing_values(
            table_name,
            df,
            validation_results
        )

        # 检查完全重复行数量
        check_duplicate_rows(
            table_name,
            df,
            validation_results
        )

        # 检查特征表行数是否和原始数据唯一 key 数一致
        check_row_count(
            table_name,
            df,
            key_cols,
            source_df,
            validation_results
        )

        # 检查数值型特征是否存在负数
        check_non_negative_columns(
            table_name,
            df,
            validation_results
        )

        # 检查比例字段是否存在负数
        check_rate_columns(
            table_name,
            df,
            validation_results
        )

    # 检查用户商品特征表中的二分类字段
    if "03_user_item_features.csv" in loaded_tables:
        check_binary_columns(
            "03_user_item_features.csv",
            loaded_tables["03_user_item_features.csv"],
            ["is_peak_hour", "is_buy"],
            validation_results
        )

    # 检查时间特征表中的二分类字段
    if "03_time_features.csv" in loaded_tables:
        check_binary_columns(
            "03_time_features.csv",
            loaded_tables["03_time_features.csv"],
            ["is_weekend", "is_workday", "is_peak_hour", "is_double12_period"],
            validation_results
        )

    # 生成 key value 说明表
    key_summary_df = create_key_summary(
        feature_config,
        loaded_tables
    )

    # 运行抽样校验
    sample_validation_df = run_sample_validation(
        loaded_tables,
        source_df
    )

    # 整理汇总校验结果
    validation_df = pd.DataFrame(validation_results)

    # 保存汇总校验结果
    validation_df.to_csv(
        OUTPUT_DIR / "03_feature_validation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 保存 key value 说明
    key_summary_df.to_csv(
        OUTPUT_DIR / "03_feature_key_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 保存抽样校验结果
    sample_validation_df.to_csv(
        OUTPUT_DIR / "03_feature_sample_validation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 统计汇总校验通过数量
    passed_count = validation_df["is_passed"].sum()
    total_count = len(validation_df)

    # 统计抽样校验通过数量
    sample_passed_count = sample_validation_df["is_passed"].sum()
    sample_total_count = len(sample_validation_df)

    # 打印汇总校验结果
    print(f"特征工程汇总校验完成：{passed_count}/{total_count} 项通过。")

    # 打印抽样校验结果
    print(
        f"特征工程抽样校验完成："
        f"{sample_passed_count}/{sample_total_count} 项通过。"
    )

    # 打印输出文件路径
    print("已保存：03_feature_validation.csv")
    print("已保存：03_feature_key_summary.csv")
    print("已保存：03_feature_sample_validation.csv")

    # 打印最终校验结论
    if passed_count == total_count and sample_passed_count == sample_total_count:
        print("所有特征工程校验通过。")
    else:
        print("部分特征工程校验未通过，请查看输出文件。")


if __name__ == "__main__":
    validate_feature_outputs()