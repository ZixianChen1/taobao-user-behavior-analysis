"""
1. 读取第三阶段生成的特征中间表
2. 以用户商品特征表作为主表
3. 按 key 合并用户特征、商品特征、类目特征、用户类目特征和近期行为特征
4. 检查合并前后行数、缺失值、重复 key 和标签分布
5. 保存建模用特征宽表和检查结果
"""

import sys
from pathlib import Path

import pandas as pd


# 将项目根目录加入 Python 搜索路径，方便读取 config.py
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))

from config import OUTPUT_DIR


def read_feature_table(file_name: str) -> pd.DataFrame:
    """
    读取特征表。
    """
    # 所有特征中间表都保存在 output_csv 文件夹中
    file_path = OUTPUT_DIR / file_name

    # 如果前一步特征工程没有生成对应文件，直接报错，避免后续合并结果不完整
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    return pd.read_csv(file_path)


def check_unique_key(df: pd.DataFrame, key_cols: list[str], table_name: str) -> None:
    """
    检查合并前 key 是否唯一。
    """
    # 如果右表 key 不唯一，merge 后可能导致宽表行数膨胀
    duplicate_count = df.duplicated(subset=key_cols).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{table_name} 的 key 不唯一，重复数量：{duplicate_count}"
        )


def add_merge_log(
    merge_logs: list[dict],
    step_name: str,
    before_rows: int,
    after_rows: int,
    before_cols: int,
    after_cols: int
) -> None:
    """
    记录每一步合并后的行数和列数变化。
    """
    merge_logs.append({
        "step_name": step_name,
        "before_rows": before_rows,
        "after_rows": after_rows,
        "row_count_changed": before_rows != after_rows,
        "before_columns": before_cols,
        "after_columns": after_cols,
        "added_columns": after_cols - before_cols
    })


def merge_feature_table(
    wide_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    merge_keys: list[str],
    table_name: str,
    merge_logs: list[dict]
) -> pd.DataFrame:
    """
    合并一张特征表，并记录合并前后变化。
    """
    # 合并前先检查右表 key，保证每个 key 只对应一行特征
    check_unique_key(feature_df, merge_keys, table_name)

    # 记录合并前行数和列数，用于后续判断 merge 是否异常
    before_rows = len(wide_df)
    before_cols = wide_df.shape[1]

    # 使用 left join，保证主表 user_id + item_id 样本不丢失
    # validate="many_to_one" 用于限制右表必须是唯一 key，防止一对多合并
    wide_df = wide_df.merge(
        feature_df,
        on=merge_keys,
        how="left",
        validate="many_to_one"
    )

    # 记录合并后行数和列数，正常情况下行数不应该变化
    after_rows = len(wide_df)
    after_cols = wide_df.shape[1]

    add_merge_log(
        merge_logs,
        f"merge_{table_name}",
        before_rows,
        after_rows,
        before_cols,
        after_cols
    )

    return wide_df


def create_wide_table_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成宽表检查汇总。
    """
    # 宽表基础规模
    total_rows = len(wide_df)
    total_columns = wide_df.shape[1]

    # 检查合并后是否产生缺失值
    total_missing_values = int(wide_df.isna().sum().sum())

    # 宽表主键应保持 user_id + item_id 唯一
    duplicate_key_count = int(
        wide_df.duplicated(subset=["user_id", "item_id"]).sum()
    )

    # is_buy 是后续建模的标签字段，这里统计正负样本分布
    if "is_buy" in wide_df.columns:
        buy_count = int((wide_df["is_buy"] == 1).sum())
        not_buy_count = int((wide_df["is_buy"] == 0).sum())
        buy_rate = buy_count / total_rows if total_rows > 0 else 0
    else:
        buy_count = 0
        not_buy_count = 0
        buy_rate = 0

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
            "description": "宽表缺失值总数"
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


def create_missing_value_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成每个字段的缺失值统计。
    """
    # 按字段统计缺失值，方便定位是哪一张特征表合并后产生缺失
    missing_df = wide_df.isna().sum().reset_index()
    missing_df.columns = ["column_name", "missing_count"]

    # 缺失比例 = 当前字段缺失数量 / 宽表总行数
    missing_df["missing_rate"] = missing_df["missing_count"] / len(wide_df)

    missing_df = missing_df.sort_values(
        by="missing_count",
        ascending=False
    )

    return missing_df


def build_feature_wide_table() -> None:
    """
    构建特征宽表。
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    merge_logs = []

    # 读取主表：一行代表一个用户和一个商品
    user_item_df = read_feature_table("03_user_item_features.csv")

    # 检查主表 key
    # 主表 key 必须唯一，否则后续模型样本会重复
    check_unique_key(
        user_item_df,
        ["user_id", "item_id"],
        "03_user_item_features.csv"
    )

    # 以用户商品特征表作为宽表主表
    # 最终宽表粒度保持为 user_id + item_id
    wide_df = user_item_df.copy()

    # 读取其他特征表
    # 这些表会按照不同 key 左连接到主表上
    user_df = read_feature_table("03_user_features.csv")
    item_df = read_feature_table("03_item_features.csv")
    category_df = read_feature_table("03_category_features.csv")
    user_category_df = read_feature_table("03_user_category_features.csv")
    recent_df = read_feature_table("03_recent_behavior_features.csv")

    # 商品表和近期行为表里也有 item_category，主表已经有了，避免重复列
    # 删除重复的 item_category，只保留主表中的类目信息
    if "item_category" in item_df.columns:
        item_df = item_df.drop(columns=["item_category"])

    if "item_category" in recent_df.columns:
        recent_df = recent_df.drop(columns=["item_category"])

    # 合并用户特征
    # key：user_id
    wide_df = merge_feature_table(
        wide_df,
        user_df,
        ["user_id"],
        "03_user_features.csv",
        merge_logs
    )

    # 合并商品特征
    # key：item_id
    wide_df = merge_feature_table(
        wide_df,
        item_df,
        ["item_id"],
        "03_item_features.csv",
        merge_logs
    )

    # 合并类目特征
    # key：item_category
    wide_df = merge_feature_table(
        wide_df,
        category_df,
        ["item_category"],
        "03_category_features.csv",
        merge_logs
    )

    # 合并用户类目特征
    # key：user_id + item_category
    wide_df = merge_feature_table(
        wide_df,
        user_category_df,
        ["user_id", "item_category"],
        "03_user_category_features.csv",
        merge_logs
    )

    # 合并近期行为特征
    # key：user_id + item_id
    wide_df = merge_feature_table(
        wide_df,
        recent_df,
        ["user_id", "item_id"],
        "03_recent_behavior_features.csv",
        merge_logs
    )

    # 生成检查结果
    # merge_log_df 记录每一步合并是否改变行数
    # summary_df 记录宽表整体情况
    # missing_df 记录每个字段的缺失情况
    merge_log_df = pd.DataFrame(merge_logs)
    summary_df = create_wide_table_summary(wide_df)
    missing_df = create_missing_value_summary(wide_df)

    # 保存宽表和检查结果
    wide_df.to_csv(
        OUTPUT_DIR / "04_feature_wide_table.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        OUTPUT_DIR / "04_feature_wide_table_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    merge_log_df.to_csv(
        OUTPUT_DIR / "04_feature_wide_table_merge_log.csv",
        index=False,
        encoding="utf-8-sig"
    )

    missing_df.to_csv(
        OUTPUT_DIR / "04_feature_wide_table_missing_values.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 打印结果
    print("特征宽表构建完成。")
    print(f"宽表行数：{wide_df.shape[0]}")
    print(f"宽表列数：{wide_df.shape[1]}")
    print(f"缺失值总数：{int(wide_df.isna().sum().sum())}")
    print(
        "重复 user_id + item_id 数量："
        f"{int(wide_df.duplicated(subset=['user_id', 'item_id']).sum())}"
    )

    if "is_buy" in wide_df.columns:
        buy_count = int((wide_df["is_buy"] == 1).sum())
        not_buy_count = int((wide_df["is_buy"] == 0).sum())
        buy_rate = buy_count / len(wide_df)

        print(f"is_buy = 0 数量：{not_buy_count}")
        print(f"is_buy = 1 数量：{buy_count}")
        print(f"购买样本比例：{buy_rate:.4%}")

    print("已保存：04_feature_wide_table.csv")
    print("已保存：04_feature_wide_table_summary.csv")
    print("已保存：04_feature_wide_table_merge_log.csv")
    print("已保存：04_feature_wide_table_missing_values.csv")


if __name__ == "__main__":
    build_feature_wide_table()
