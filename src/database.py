"""
数据库模块：将清洗后的数据保存到 SQLite 数据库。
"""

import sqlite3
from pathlib import Path

import pandas as pd


def save_dataframe_to_sqlite(
    df: pd.DataFrame,
    db_path: Path,
    table_name: str
) -> None:
    """
    将 DataFrame 保存到 SQLite 数据库。
    """
    try:
        # 连接 SQLite 数据库
        conn = sqlite3.connect(db_path)

        # 将 DataFrame 写入数据库表
        df.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False
        )

        # 创建 cursor，用来执行 SQL 语句
        cursor = conn.cursor()

        # 为 user_id 创建索引
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_user_id "
            f"ON {table_name}(user_id);"
        )

        # 为 item_id 创建索引
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_item_id "
            f"ON {table_name}(item_id);"
        )

        # 为 item_category 创建索引
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_item_category "
            f"ON {table_name}(item_category);"
        )

        # 为 behavior_type 创建索引
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_behavior_type "
            f"ON {table_name}(behavior_type);"
        )

        # 为 time 创建索引
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_time "
            f"ON {table_name}(time);"
        )

        # 提交数据库修改
        conn.commit()

        print(f"数据已成功保存到 SQLite：{db_path}")
        print(f"数据库表名：{table_name}")
        print(f"保存行数：{len(df)}")

    except Exception as error:
        # 保存数据库出错时提示错误
        print(f"保存 SQLite 数据库时发生错误：{error}")
        raise

    finally:
        # 关闭数据库连接
        if "conn" in locals():
            conn.close()