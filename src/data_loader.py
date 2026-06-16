"""
数据读取模块：负责读取 CSV 数据。
"""

from pathlib import Path

import pandas as pd


def load_data(file_path: Path) -> pd.DataFrame:
    """
    读取 CSV 文件，并返回 DataFrame。
    """
    try:
        # 读取 CSV 数据
        df = pd.read_csv(file_path)

        # 打印数据行数和列数
        print(f"数据读取成功，数据维度为：{df.shape}")

        # 返回读取后的数据表
        return df

    except FileNotFoundError:
        # 文件不存在时提示路径问题
        print(f"文件不存在，请检查路径：{file_path}")
        raise

    except Exception as error:
        # 其他读取错误
        print(f"读取数据时发生错误：{error}")
        raise