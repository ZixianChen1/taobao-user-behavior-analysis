"""
第七阶段：核心特征筛选和时间窗口数据集划分。

作用：
1. 读取第六阶段生成的 Parquet 建模数据集
2. 获取每个 user_id + item_id 样本的最后行为时间
3. 按时间顺序划分训练集、验证集和测试集，减少时间泄露风险
4. 只在训练集上计算特征和标签的相关性
5. 根据相关性筛选核心特征
6. 保存划分后的数据和检查结果为 Parquet
"""

import sys
from pathlib import Path

import pandas as pd


# 将项目根目录加入 Python 搜索路径，方便读取 config.py
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))

from config import DATA_FILE, OUTPUT_DIR, TIME_FORMAT


# Parquet 输出文件夹
PARQUET_DIR = BASE_PATH / "output_parquet"

MODEL_DATASET_FILE = "06_model_dataset.parquet"
WIDE_TABLE_PARQUET_FILE = "04_feature_wide_table.parquet"
WIDE_TABLE_CSV_FILE = "04_feature_wide_table.csv"

LABEL_COLUMN = "is_buy"

# 最多保留多少个核心特征
TOP_FEATURE_COUNT = 30

# 相关性计算最多抽样多少行，避免运行太慢
CORRELATION_SAMPLE_SIZE = 500000

# 时间窗口划分比例
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2

RANDOM_STATE = 42


def read_model_dataset() -> pd.DataFrame:
    """
    读取第六阶段建模数据集。
    """
    file_path = PARQUET_DIR / MODEL_DATASET_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    return pd.read_parquet(file_path, engine="pyarrow")


def read_wide_table() -> pd.DataFrame:
    """
    读取第四阶段特征宽表。
    优先读取 Parquet，如果没有则读取 CSV。
    """
    parquet_path = PARQUET_DIR / WIDE_TABLE_PARQUET_FILE
    csv_path = OUTPUT_DIR / WIDE_TABLE_CSV_FILE

    if parquet_path.exists():
        return pd.read_parquet(parquet_path, engine="pyarrow")

    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        f"宽表文件不存在：{parquet_path} 或 {csv_path}"
    )


def create_last_behavior_time_from_raw(wide_df: pd.DataFrame) -> pd.Series:
    """
    如果宽表中没有 last_behavior_time，则从原始数据重新计算。
    """
    print("宽表中没有 last_behavior_time，开始从原始数据重新计算。")

    raw_df = pd.read_csv(
        DATA_FILE,
        usecols=["time", "user_id", "item_id", "behavior_type"]
    )

    raw_df["time"] = pd.to_datetime(
        raw_df["time"],
        format=TIME_FORMAT,
        errors="coerce"
    )

    raw_df = raw_df.dropna(subset=["time"]).copy()

    last_time_df = (
        raw_df
        .groupby(["user_id", "item_id"], as_index=False)
        .agg(last_behavior_time=("time", "max"))
    )

    time_df = wide_df[["user_id", "item_id"]].merge(
        last_time_df,
        on=["user_id", "item_id"],
        how="left",
        validate="many_to_one"
    )

    return time_df["last_behavior_time"]


def get_split_time(wide_df: pd.DataFrame) -> pd.Series:
    """
    获取用于时间划分的时间字段。
    """
    if "last_behavior_time" in wide_df.columns:
        split_time = pd.to_datetime(
            wide_df["last_behavior_time"],
            errors="coerce"
        )
        time_source = "last_behavior_time_from_wide_table"

    else:
        split_time = create_last_behavior_time_from_raw(wide_df)
        split_time = pd.to_datetime(split_time, errors="coerce")
        time_source = "last_behavior_time_recomputed_from_raw_data"

    missing_time_count = int(split_time.isna().sum())

    if missing_time_count > 0:
        raise ValueError(f"存在无法获取时间的样本，数量：{missing_time_count}")

    print(f"时间划分字段来源：{time_source}")

    return split_time, time_source


def remove_low_value_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    删除低价值字段。
    """
    feature_df = df.drop(columns=[LABEL_COLUMN])
    remove_records = []

    # 只保留数值型字段，方便后续 baseline 模型训练
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [
        col for col in feature_df.columns
        if col not in numeric_cols
    ]

    for col in non_numeric_cols:
        remove_records.append({
            "feature_name": col,
            "remove_reason": "非数值型字段，暂不进入 baseline 模型"
        })

    feature_df = feature_df[numeric_cols].copy()

    # 删除只有一个取值的字段
    single_value_cols = [
        col for col in feature_df.columns
        if feature_df[col].nunique(dropna=False) <= 1
    ]

    for col in single_value_cols:
        remove_records.append({
            "feature_name": col,
            "remove_reason": "字段只有一个取值，对模型没有区分作用"
        })

    feature_df = feature_df.drop(columns=single_value_cols)

    remove_summary_df = pd.DataFrame(remove_records)

    return feature_df, remove_summary_df


def split_dataset_by_time(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    split_time: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    按时间顺序划分训练集、验证集和测试集。
    """
    split_df = pd.DataFrame({
        "split_time": split_time,
        LABEL_COLUMN: target_series
    })

    # 按时间从早到晚排序
    sorted_index = split_df.sort_values("split_time").index.tolist()

    total_count = len(sorted_index)
    train_end = int(total_count * TRAIN_RATIO)
    valid_end = int(total_count * (TRAIN_RATIO + VALID_RATIO))

    train_index = sorted_index[:train_end]
    valid_index = sorted_index[train_end:valid_end]
    test_index = sorted_index[valid_end:]

    x_train = feature_df.loc[train_index].copy()
    x_valid = feature_df.loc[valid_index].copy()
    x_test = feature_df.loc[test_index].copy()

    y_train = target_series.loc[train_index].copy()
    y_valid = target_series.loc[valid_index].copy()
    y_test = target_series.loc[test_index].copy()

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def calculate_feature_correlation(
    x_train: pd.DataFrame,
    y_train: pd.Series
) -> pd.DataFrame:
    """
    只在训练集上计算特征和标签的相关性。
    """
    if len(x_train) > CORRELATION_SAMPLE_SIZE:
        sample_x = x_train.sample(
            n=CORRELATION_SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )
        sample_y = y_train.loc[sample_x.index]
    else:
        sample_x = x_train
        sample_y = y_train

    records = []

    for col in sample_x.columns:
        corr_value = sample_x[col].corr(sample_y)

        if pd.isna(corr_value):
            corr_value = 0

        records.append({
            "feature_name": col,
            "correlation_with_label": corr_value,
            "abs_correlation": abs(corr_value)
        })

    correlation_df = pd.DataFrame(records)

    correlation_df = correlation_df.sort_values(
        by="abs_correlation",
        ascending=False
    )

    return correlation_df


def select_core_features(correlation_df: pd.DataFrame) -> list[str]:
    """
    根据训练集相关性选择核心特征。
    """
    return (
        correlation_df
        .head(TOP_FEATURE_COUNT)["feature_name"]
        .tolist()
    )


def create_split_summary(
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    生成正负样本分布检查结果。
    """
    records = []

    for dataset_name, y_data in [
        ("train", y_train),
        ("valid", y_valid),
        ("test", y_test)
    ]:
        total_count = len(y_data)
        buy_count = int((y_data == 1).sum())
        not_buy_count = int((y_data == 0).sum())
        buy_rate = buy_count / total_count if total_count > 0 else 0

        records.append({
            "dataset": dataset_name,
            "total_count": total_count,
            "not_buy_count": not_buy_count,
            "buy_count": buy_count,
            "buy_rate": buy_rate
        })

    return pd.DataFrame(records)


def create_time_split_summary(
    split_time: pd.Series,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    生成时间窗口划分检查结果。
    """
    records = []

    for dataset_name, y_data in [
        ("train", y_train),
        ("valid", y_valid),
        ("test", y_test)
    ]:
        dataset_time = split_time.loc[y_data.index]

        records.append({
            "dataset": dataset_name,
            "start_time": dataset_time.min(),
            "end_time": dataset_time.max(),
            "sample_count": len(y_data)
        })

    return pd.DataFrame(records)


def save_dataset(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    file_name: str
) -> None:
    """
    保存带标签的数据集为 Parquet。
    """
    output_df = x_data.copy()
    output_df[LABEL_COLUMN] = y_data.values

    output_df.to_parquet(
        PARQUET_DIR / file_name,
        index=False,
        engine="pyarrow"
    )


def save_parquet(df: pd.DataFrame, file_name: str) -> None:
    """
    保存普通结果表为 Parquet。
    """
    df.to_parquet(
        PARQUET_DIR / file_name,
        index=False,
        engine="pyarrow"
    )


def run_feature_selection_and_split() -> None:
    """
    运行核心特征筛选和时间窗口数据集划分。
    """
    PARQUET_DIR.mkdir(exist_ok=True)

    model_df = read_model_dataset()
    wide_df = read_wide_table()

    if LABEL_COLUMN not in model_df.columns:
        raise ValueError(f"建模数据集中不存在标签字段：{LABEL_COLUMN}")

    if len(model_df) != len(wide_df):
        raise ValueError(
            "06 建模数据集和 04 宽表行数不一致，不能对齐时间字段。"
        )

    split_time, time_source = get_split_time(wide_df)

    target_series = model_df[LABEL_COLUMN]

    # 删除低价值字段
    feature_df, remove_summary_df = remove_low_value_features(model_df)

    # 先按时间划分，防止未来信息进入训练集
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_dataset_by_time(
        feature_df,
        target_series,
        split_time
    )

    # 只在训练集上做相关性筛选，避免用到验证集和测试集信息
    correlation_df = calculate_feature_correlation(
        x_train,
        y_train
    )

    core_features = select_core_features(correlation_df)

    x_train_core = x_train[core_features].copy()
    x_valid_core = x_valid[core_features].copy()
    x_test_core = x_test[core_features].copy()

    split_summary_df = create_split_summary(
        y_train,
        y_valid,
        y_test
    )

    time_split_summary_df = create_time_split_summary(
        split_time,
        y_train,
        y_valid,
        y_test
    )

    core_feature_list_df = pd.DataFrame({
        "feature_name": core_features,
        "rank": range(1, len(core_features) + 1)
    })

    core_feature_list_df = core_feature_list_df.merge(
        correlation_df,
        on="feature_name",
        how="left"
    )

    time_source_df = pd.DataFrame([{
        "time_source": time_source,
        "description": "用于按时间顺序划分训练集、验证集和测试集"
    }])

    # 保存划分后的数据集
    save_dataset(x_train_core, y_train, "07_train_dataset.parquet")
    save_dataset(x_valid_core, y_valid, "07_valid_dataset.parquet")
    save_dataset(x_test_core, y_test, "07_test_dataset.parquet")

    # 保存检查结果
    save_parquet(correlation_df, "07_feature_correlation.parquet")
    save_parquet(core_feature_list_df, "07_core_feature_list.parquet")
    save_parquet(remove_summary_df, "07_removed_features.parquet")
    save_parquet(split_summary_df, "07_dataset_split_summary.parquet")
    save_parquet(time_split_summary_df, "07_time_split_summary.parquet")
    save_parquet(time_source_df, "07_time_source.parquet")

    print("核心特征筛选和时间窗口数据集划分完成。")
    print(f"时间字段来源：{time_source}")
    print(f"原始建模数据行数：{model_df.shape[0]}")
    print(f"原始候选特征数量：{model_df.shape[1] - 1}")
    print(f"筛选后核心特征数量：{len(core_features)}")
    print(f"训练集行数：{len(x_train_core)}")
    print(f"验证集行数：{len(x_valid_core)}")
    print(f"测试集行数：{len(x_test_core)}")

    print("已保存：07_train_dataset.parquet")
    print("已保存：07_valid_dataset.parquet")
    print("已保存：07_test_dataset.parquet")
    print("已保存：07_feature_correlation.parquet")
    print("已保存：07_core_feature_list.parquet")
    print("已保存：07_removed_features.parquet")
    print("已保存：07_dataset_split_summary.parquet")
    print("已保存：07_time_split_summary.parquet")
    print("已保存：07_time_source.parquet")


if __name__ == "__main__":
    run_feature_selection_and_split()