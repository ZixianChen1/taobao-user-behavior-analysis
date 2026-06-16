"""
结果校验模块：检查 output_csv 中各个统计结果是否一致。
"""

from pathlib import Path

import pandas as pd


def validate_outputs(output_dir: Path) -> None:
    """
    校验基础统计输出结果是否一致。
    """
    # 读取整体数据概览
    overall_summary = pd.read_csv(output_dir / "01_overall_summary.csv")

    # 读取行为分布结果
    behavior_distribution = pd.read_csv(
        output_dir / "01_behavior_distribution.csv"
    )

    # 读取用户互动次数结果
    user_interaction_count = pd.read_csv(
        output_dir / "01_user_interaction_count.csv"
    )

    # 读取商品互动次数结果
    item_interaction_count = pd.read_csv(
        output_dir / "01_item_interaction_count.csv"
    )

    # 读取类目统计结果
    category_summary = pd.read_csv(output_dir / "01_category_summary.csv")

    # 读取每日行为统计结果
    daily_behavior_count = pd.read_csv(
        output_dir / "01_daily_behavior_count.csv"
    )

    # 读取每小时行为统计结果
    hourly_behavior_count = pd.read_csv(
        output_dir / "01_hourly_behavior_count.csv"
    )

    # 读取转化率统计结果
    conversion_summary = pd.read_csv(
        output_dir / "01_conversion_summary.csv"
    )

    # 读取数据合理性检查结果
    validity_check = pd.read_csv(output_dir / "01_validity_check.csv")

    # 取出总记录数
    total_records = int(
        overall_summary.loc[
            overall_summary["metric"] == "total_records",
            "value"
        ].iloc[0]
    )

    # 计算各统计文件中的总数
    behavior_total = int(behavior_distribution["behavior_count"].sum())

    user_total = int(
        user_interaction_count["total_interaction_count"].sum()
    )

    item_total = int(
        item_interaction_count["total_interaction_count"].sum()
    )

    category_total = int(category_summary["total_interaction_count"].sum())

    daily_total = int(daily_behavior_count["behavior_count"].sum())

    hourly_total = int(hourly_behavior_count["behavior_count"].sum())

    # 转化率文件中的行为数量
    conversion_dict = dict(
        zip(conversion_summary["metric"], conversion_summary["value"])
    )

    pv_count = int(conversion_dict.get("pv_count", 0))
    fav_count = int(conversion_dict.get("fav_count", 0))
    cart_count = int(conversion_dict.get("cart_count", 0))
    buy_count = int(conversion_dict.get("buy_count", 0))

    # 行为分布文件中的行为数量
    behavior_dict = dict(
        zip(
            behavior_distribution["behavior_name"],
            behavior_distribution["behavior_count"]
        )
    )

    # 合理性检查结果
    validity_dict = dict(
        zip(validity_check["check_item"], validity_check["result"])
    )

    # 整理校验结果
    validation_results = pd.DataFrame({
        "check_item": [
            "behavior_total_equals_total_records",
            "user_total_equals_total_records",
            "item_total_equals_total_records",
            "category_total_equals_total_records",
            "daily_total_equals_total_records",
            "hourly_total_equals_total_records",
            "pv_count_matches_behavior_distribution",
            "fav_count_matches_behavior_distribution",
            "cart_count_matches_behavior_distribution",
            "buy_count_matches_behavior_distribution",
            "invalid_behavior_type_count_is_zero",
            "invalid_time_count_is_zero",
            "invalid_hour_count_is_zero",
            "invalid_date_count_is_zero"
        ],
        "expected": [
            total_records,
            total_records,
            total_records,
            total_records,
            total_records,
            total_records,
            int(behavior_dict.get("pv", 0)),
            int(behavior_dict.get("fav", 0)),
            int(behavior_dict.get("cart", 0)),
            int(behavior_dict.get("buy", 0)),
            0,
            0,
            0,
            0
        ],
        "actual": [
            behavior_total,
            user_total,
            item_total,
            category_total,
            daily_total,
            hourly_total,
            pv_count,
            fav_count,
            cart_count,
            buy_count,
            int(validity_dict.get("invalid_behavior_type_count", 0)),
            int(validity_dict.get("invalid_time_count", 0)),
            int(validity_dict.get("invalid_hour_count", 0)),
            int(validity_dict.get("invalid_date_count", 0))
        ]
    })

    # 判断是否通过
    validation_results["is_passed"] = (
        validation_results["expected"] == validation_results["actual"]
    )

    # 保存结果校验文件
    validation_results.to_csv(
        output_dir / "01_output_validation.csv",
        index=False
    )

    # 打印校验结果
    passed_count = validation_results["is_passed"].sum()
    total_count = len(validation_results)

    print(f"结果校验完成：{passed_count}/{total_count} 项通过。")

    if passed_count == total_count:
        print("所有输出结果校验通过。")
    else:
        print("部分输出结果未通过，请查看 01_output_validation.csv。")