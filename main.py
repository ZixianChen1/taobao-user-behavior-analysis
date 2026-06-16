"""
项目主运行文件：串联读取、检查、清洗、入库、基础统计和结果校验流程。
"""

from config import (
    BEHAVIOR_MAP,
    DATA_FILE,
    DATABASE_DIR,
    DATABASE_PATH,
    OUTPUT_DIR,
    SQL_TABLE_NAME,
    TIME_FORMAT
)
from src.basic_statistics import (
    save_behavior_distribution,
    save_category_statistics,
    save_conversion_statistics,
    save_item_statistics,
    save_overall_summary,
    save_time_statistics,
    save_user_statistics
)
from src.data_check import (
    print_basic_info,
    save_data_types,
    save_duplicate_check,
    save_missing_values,
    save_unique_counts,
    save_validity_check
)
from src.data_cleaning import clean_data
from src.data_loader import load_data
from src.database import save_dataframe_to_sqlite
from src.result_validation import validate_outputs


def main() -> None:
    """
    运行第一阶段数据处理、基础统计和结果校验。
    """
    # 创建输出文件夹
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 创建数据库文件夹
    DATABASE_DIR.mkdir(exist_ok=True)

    # 读取原始数据
    raw_df = load_data(DATA_FILE)

    # 打印基础信息
    print_basic_info(raw_df)

    # 保存数据类型检查结果
    save_data_types(raw_df, OUTPUT_DIR)

    # 保存缺失值检查结果
    save_missing_values(raw_df, OUTPUT_DIR)

    # 保存唯一值数量检查结果
    save_unique_counts(raw_df, OUTPUT_DIR)

    # 保存重复行检查结果
    save_duplicate_check(raw_df, OUTPUT_DIR)

    # 保存数据合理性检查结果
    # 检查 behavior_type 是否只有 1、2、3、4
    # 检查 time 是否存在非法日期或非法小时
    save_validity_check(
        raw_df,
        OUTPUT_DIR,
        valid_behavior_types=list(BEHAVIOR_MAP.keys()),
        time_format=TIME_FORMAT
    )

    # 清洗和预处理数据
    cleaned_df = clean_data(
        raw_df,
        behavior_map=BEHAVIOR_MAP,
        time_format=TIME_FORMAT
    )

    # 保存清洗后的数据到 SQLite
    save_dataframe_to_sqlite(
        cleaned_df,
        db_path=DATABASE_PATH,
        table_name=SQL_TABLE_NAME
    )

    # 保存整体数据概览
    save_overall_summary(cleaned_df, OUTPUT_DIR)

    # 保存行为类型分布
    save_behavior_distribution(cleaned_df, OUTPUT_DIR)

    # 保存用户维度统计
    save_user_statistics(cleaned_df, OUTPUT_DIR)

    # 保存商品维度统计
    save_item_statistics(cleaned_df, OUTPUT_DIR)

    # 保存类目维度统计
    save_category_statistics(cleaned_df, OUTPUT_DIR)

    # 保存时间维度统计
    save_time_statistics(cleaned_df, OUTPUT_DIR)

    # 保存转化率统计
    save_conversion_statistics(cleaned_df, OUTPUT_DIR)

    # 校验 output_csv 中各个统计结果是否一致
    validate_outputs(OUTPUT_DIR)

    print("\n第一阶段数据处理、基础统计和结果校验已全部完成。")


# 直接运行 main.py 时执行 main()
if __name__ == "__main__":
    main()