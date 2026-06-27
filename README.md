# Taobao User Behavior Analysis

根据淘宝用户数据，进行了数据检查、数据清洗、基础统计、可视化分析和特征工程，为后续用户购买行为预测做准备。

## 项目结构

```text
Alibaba/
├── data/                         # 原始数据文件
├── database/                     # SQLite 数据库文件
├── figures/                      # 可视化图表
├── output_csv/                   # 数据检查、统计和特征工程输出结果
├── sql_scripts/                  # SQL 脚本
├── feature_engineering/          # 特征说明文件
├── src/                          # Python 功能模块
├── config.py                     # 项目配置文件
├── main.py                       # 数据处理和基础统计主程序
├── 02_visualization.py           # 可视化分析脚本
└── requirements.txt              # Python 依赖包