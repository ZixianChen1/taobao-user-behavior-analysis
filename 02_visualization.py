"""
第二阶段可视化脚本。

作用：
1. 读取 output_csv 中的基础统计结果
2. 生成可视化图表
3. 保存部分二次汇总结果，支撑报告中的具体结论
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from config import BASE_DIR, OUTPUT_DIR


FIGURE_DIR = BASE_DIR / "figures"


def format_number(x, pos) -> str:
    """
    坐标轴数字转为千分位格式，避免显示 1e6。
    """
    return f"{int(x):,}"


def save_plot(file_name: str) -> None:
    """
    保存图表到 figures 文件夹。
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / file_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_behavior_distribution() -> None:
    """
    绘制行为类型分布柱状图。
    """
    df = pd.read_csv(OUTPUT_DIR / "01_behavior_distribution.csv")

    plt.figure(figsize=(8, 5))
    plt.bar(df["behavior_name"], df["behavior_count"])

    plt.title("Behavior Type Distribution")
    plt.xlabel("Behavior Type")
    plt.ylabel("Behavior Count")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_number))

    for index, value in enumerate(df["behavior_count"]):
        plt.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8)

    save_plot("02_behavior_distribution.png")


def plot_conversion_rates() -> None:
    """
    绘制转化率对比图。
    """
    df = pd.read_csv(OUTPUT_DIR / "01_conversion_summary.csv")

    ratio_df = df[df["metric"].isin([
        "buy_pv_ratio",
        "buy_cart_ratio",
        "buy_fav_ratio"
    ])].copy()

    # 小数转百分比
    ratio_df["value"] = ratio_df["value"] * 100

    label_map = {
        "buy_pv_ratio": "Buy / PV",
        "buy_cart_ratio": "Buy / Cart",
        "buy_fav_ratio": "Buy / Fav"
    }
    ratio_df["metric"] = ratio_df["metric"].map(label_map)

    plt.figure(figsize=(8, 5))
    plt.bar(ratio_df["metric"], ratio_df["value"])

    plt.title("Conversion Rate Comparison")
    plt.xlabel("Conversion Metric")
    plt.ylabel("Conversion Rate (%)")

    for index, value in enumerate(ratio_df["value"]):
        plt.text(index, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)

    save_plot("02_conversion_rates.png")


def plot_daily_behavior_trend() -> None:
    """
    绘制每日总行为趋势图。
    """
    df = pd.read_csv(OUTPUT_DIR / "01_daily_behavior_count.csv")

    # 每日总行为量 = 当天 pv、fav、cart、buy 数量之和
    daily_total = (
        df
        .groupby("date", as_index=False)["behavior_count"]
        .sum()
        .sort_values("date")
    )

    daily_total.to_csv(
        OUTPUT_DIR / "02_daily_total_behavior.csv",
        index=False
    )

    plt.figure(figsize=(12, 5))
    plt.plot(daily_total["date"], daily_total["behavior_count"], marker="o")

    plt.title("Daily Total Behavior Trend")
    plt.xlabel("Date")
    plt.ylabel("Behavior Count")
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_number))

    save_plot("02_daily_total_behavior_trend.png")


def plot_daily_buy_trend() -> None:
    """
    绘制每日购买行为趋势图。
    """
    df = pd.read_csv(OUTPUT_DIR / "01_daily_behavior_count.csv")

    daily_buy = (
        df[df["behavior_name"] == "buy"]
        .groupby("date", as_index=False)["behavior_count"]
        .sum()
        .sort_values("date")
    )

    daily_buy.to_csv(
        OUTPUT_DIR / "02_daily_buy_behavior.csv",
        index=False
    )

    plt.figure(figsize=(12, 5))
    plt.plot(daily_buy["date"], daily_buy["behavior_count"], marker="o")

    plt.title("Daily Buy Behavior Trend")
    plt.xlabel("Date")
    plt.ylabel("Buy Count")
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_number))

    save_plot("02_daily_buy_behavior_trend.png")


def plot_hourly_behavior_trend() -> None:
    """
    绘制每小时总行为趋势图。
    """
    df = pd.read_csv(OUTPUT_DIR / "01_hourly_behavior_count.csv")

    # 每小时总行为量 = 该小时 pv、fav、cart、buy 数量之和
    hourly_total = (
        df
        .groupby("hour", as_index=False)["behavior_count"]
        .sum()
        .sort_values("hour")
    )

    hourly_total.to_csv(
        OUTPUT_DIR / "02_hourly_total_behavior.csv",
        index=False
    )

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_total["hour"], hourly_total["behavior_count"], marker="o")

    plt.title("Hourly Total Behavior Trend")
    plt.xlabel("Hour")
    plt.ylabel("Behavior Count")
    plt.xticks(range(0, 24))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_number))

    save_plot("02_hourly_total_behavior_trend.png")


def plot_hourly_buy_trend() -> None:
    """
    绘制每小时购买行为趋势图。
    """
    df = pd.read_csv(OUTPUT_DIR / "01_hourly_behavior_count.csv")

    hourly_buy = (
        df[df["behavior_name"] == "buy"]
        .groupby("hour", as_index=False)["behavior_count"]
        .sum()
        .sort_values("hour")
    )

    hourly_buy.to_csv(
        OUTPUT_DIR / "02_hourly_buy_behavior.csv",
        index=False
    )

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_buy["hour"], hourly_buy["behavior_count"], marker="o")

    plt.title("Hourly Buy Behavior Trend")
    plt.xlabel("Hour")
    plt.ylabel("Buy Count")
    plt.xticks(range(0, 24))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_number))

    save_plot("02_hourly_buy_behavior_trend.png")


def plot_top_category_interaction() -> None:
    """
    绘制总互动次数最高的 Top10 类目。
    """
    df = pd.read_csv(OUTPUT_DIR / "01_category_summary.csv")

    top_category = (
        df
        .sort_values("total_interaction_count", ascending=False)
        .head(10)
        .copy()
    )

    top_category["item_category"] = top_category["item_category"].astype(str)

    top_category.to_csv(
        OUTPUT_DIR / "02_top10_category_interaction.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))
    plt.barh(top_category["item_category"], top_category["total_interaction_count"])

    plt.title("Top 10 Categories by Total Interaction")
    plt.xlabel("Total Interaction Count")
    plt.ylabel("Item Category")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_number))
    plt.gca().invert_yaxis()

    save_plot("02_top10_category_interaction.png")


def plot_top_category_buy() -> None:
    """
    绘制购买次数最高的 Top10 类目。
    """
    df = pd.read_csv(OUTPUT_DIR / "01_category_behavior_counts.csv")

    if "buy" not in df.columns:
        print("01_category_behavior_counts.csv 中没有 buy 列，请检查数据。")
        return

    top_buy_category = (
        df
        .sort_values("buy", ascending=False)
        .head(10)
        .copy()
    )

    top_buy_category["item_category"] = top_buy_category["item_category"].astype(str)

    top_buy_category.to_csv(
        OUTPUT_DIR / "02_top10_category_buy.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))
    plt.barh(top_buy_category["item_category"], top_buy_category["buy"])

    plt.title("Top 10 Categories by Buy Count")
    plt.xlabel("Buy Count")
    plt.ylabel("Item Category")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_number))
    plt.gca().invert_yaxis()

    save_plot("02_top10_category_buy.png")


def plot_user_interaction_distribution() -> None:
    """
    绘制用户互动次数分布图。
    """
    df = pd.read_csv(OUTPUT_DIR / "01_user_interaction_count.csv")

    # 只展示 95% 分位数以内的数据，避免极端值影响图形
    upper_limit = df["total_interaction_count"].quantile(0.95)

    plt.figure(figsize=(10, 5))
    plt.hist(
        df["total_interaction_count"],
        bins=50,
        range=(0, upper_limit)
    )

    plt.title("User Interaction Count Distribution")
    plt.xlabel("Total Interaction Count")
    plt.ylabel("User Count")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_number))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_number))

    save_plot("02_user_interaction_distribution.png")


def plot_top_item_interaction() -> None:
    """
    绘制总互动次数最高的 Top20 商品。
    """
    df = pd.read_csv(OUTPUT_DIR / "01_item_interaction_count.csv")

    top_items = (
        df
        .sort_values("total_interaction_count", ascending=False)
        .head(20)
        .copy()
    )

    top_items["item_id"] = top_items["item_id"].astype(str)

    top_items.to_csv(
        OUTPUT_DIR / "02_top20_item_interaction.csv",
        index=False
    )

    plt.figure(figsize=(10, 7))
    plt.barh(top_items["item_id"], top_items["total_interaction_count"])

    plt.title("Top 20 Items by Total Interaction")
    plt.xlabel("Total Interaction Count")
    plt.ylabel("Item ID")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_number))
    plt.gca().invert_yaxis()

    save_plot("02_top20_item_interaction.png")


def save_key_findings() -> None:
    """
    保存报告中用到的关键发现。
    """
    daily_total = pd.read_csv(OUTPUT_DIR / "02_daily_total_behavior.csv")
    hourly_total = pd.read_csv(OUTPUT_DIR / "02_hourly_total_behavior.csv")
    hourly_buy = pd.read_csv(OUTPUT_DIR / "02_hourly_buy_behavior.csv")
    top_category = pd.read_csv(OUTPUT_DIR / "02_top10_category_interaction.csv")
    top_buy_category = pd.read_csv(OUTPUT_DIR / "02_top10_category_buy.csv")

    findings = []

    for _, row in daily_total.sort_values("behavior_count", ascending=False).head(3).iterrows():
        findings.append({
            "finding_type": "top_daily_behavior",
            "name": row["date"],
            "value": int(row["behavior_count"])
        })

    for _, row in hourly_total.sort_values("behavior_count", ascending=False).head(3).iterrows():
        findings.append({
            "finding_type": "top_hourly_behavior",
            "name": int(row["hour"]),
            "value": int(row["behavior_count"])
        })

    for _, row in hourly_buy.sort_values("behavior_count", ascending=False).head(3).iterrows():
        findings.append({
            "finding_type": "top_hourly_buy",
            "name": int(row["hour"]),
            "value": int(row["behavior_count"])
        })

    for _, row in top_category.head(5).iterrows():
        findings.append({
            "finding_type": "top_category_interaction",
            "name": row["item_category"],
            "value": int(row["total_interaction_count"])
        })

    for _, row in top_buy_category.head(5).iterrows():
        findings.append({
            "finding_type": "top_category_buy",
            "name": row["item_category"],
            "value": int(row["buy"])
        })

    pd.DataFrame(findings).to_csv(
        OUTPUT_DIR / "02_key_findings.csv",
        index=False
    )

    print("关键发现结果已保存：02_key_findings.csv")


def main() -> None:
    """
    运行可视化流程。
    """
    FIGURE_DIR.mkdir(exist_ok=True)

    plot_behavior_distribution()
    plot_conversion_rates()
    plot_daily_behavior_trend()
    plot_daily_buy_trend()
    plot_hourly_behavior_trend()
    plot_hourly_buy_trend()
    plot_top_category_interaction()
    plot_top_category_buy()
    plot_user_interaction_distribution()
    plot_top_item_interaction()
    save_key_findings()

    print("第二阶段可视化图表已全部生成。")
    print(f"图表保存路径：{FIGURE_DIR}")


if __name__ == "__main__":
    main()