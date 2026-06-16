"""
项目配置文件：集中存放路径、数据库、表名、行为映射和时间格式。
"""

from pathlib import Path


# 当前项目根目录，也就是 Alibaba 文件夹
BASE_DIR = Path(__file__).resolve().parent

# 原始数据文件夹
DATA_DIR = BASE_DIR / "data"

# 结果输出文件夹
OUTPUT_DIR = BASE_DIR / "output_csv"

# 数据库文件夹
DATABASE_DIR = BASE_DIR / "database"

# SQL 脚本文件夹
SQL_DIR = BASE_DIR / "sql_scripts"

# 原始数据文件路径，使用相对路径，避免写死电脑路径
DATA_FILE = DATA_DIR / "taobao_user_behavior_processed.csv"

# SQLite 数据库文件路径
DATABASE_PATH = DATABASE_DIR / "taobao_behavior.db"

# SQLite 中保存清洗后数据的表名
SQL_TABLE_NAME = "user_behavior"

# 用户行为类型映射
# 1 浏览，2 收藏，3 加购，4 购买
BEHAVIOR_MAP = {
    1: "pv",
    2: "fav",
    3: "cart",
    4: "buy"
}

# time 字段格式，例如 2017-11-25 13
TIME_FORMAT = "%Y-%m-%d %H"