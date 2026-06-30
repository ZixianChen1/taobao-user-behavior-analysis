"""
特征工程性能检查脚本。

作用：
1. 运行特征工程脚本
2. 运行特征工程校验脚本
3. 记录运行时间
4. 记录内存峰值
5. 输出性能检查结果
"""

import sys
import time
import threading
from pathlib import Path

import pandas as pd


# 获取项目根目录
BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))

from config import OUTPUT_DIR
from src.feature_engineering import main as run_feature_engineering
from src.feature_validation import validate_feature_outputs


try:
    import psutil
except ImportError:
    psutil = None


def monitor_memory(
    stop_event: threading.Event,
    memory_records: list[float],
    interval_seconds: float = 0.5
) -> None:
    """
    定时记录当前进程内存占用。
    """
    if psutil is None:
        return

    process = psutil.Process()

    while not stop_event.is_set():
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_records.append(memory_mb)
        time.sleep(interval_seconds)


def run_with_performance_monitor(step_name: str, func) -> dict:
    """
    运行指定函数，并记录耗时和内存峰值。
    """
    memory_records = []
    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=monitor_memory,
        args=(stop_event, memory_records),
        daemon=True
    )

    start_time = time.perf_counter()
    monitor_thread.start()

    status = "success"
    error_message = ""

    try:
        func()
    except Exception as error:
        status = "failed"
        error_message = str(error)
    finally:
        stop_event.set()
        monitor_thread.join()

    end_time = time.perf_counter()

    elapsed_seconds = end_time - start_time
    peak_memory_mb = max(memory_records) if memory_records else None

    return {
        "step_name": step_name,
        "status": status,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "elapsed_minutes": round(elapsed_seconds / 60, 2),
        "peak_memory_mb": round(peak_memory_mb, 2) if peak_memory_mb else None,
        "peak_memory_gb": round(peak_memory_mb / 1024, 2) if peak_memory_mb else None,
        "error_message": error_message
    }


def main() -> None:
    """
    运行特征工程和校验性能检查。
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    if psutil is None:
        print("提示：当前环境未安装 psutil，只能记录运行时间，不能记录内存峰值。")
        print("如需记录内存峰值，请先运行：pip install psutil")

    performance_results = []

    print("开始运行特征工程性能检查。")

    feature_engineering_result = run_with_performance_monitor(
        "feature_engineering",
        run_feature_engineering
    )
    performance_results.append(feature_engineering_result)

    print("特征工程运行完成。")

    feature_validation_result = run_with_performance_monitor(
        "feature_validation",
        validate_feature_outputs
    )
    performance_results.append(feature_validation_result)

    print("特征工程校验运行完成。")

    performance_df = pd.DataFrame(performance_results)

    performance_df.to_csv(
        OUTPUT_DIR / "03_feature_performance_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("性能检查完成。")
    print(performance_df)
    print("已保存：03_feature_performance_summary.csv")


if __name__ == "__main__":
    main()