from pathlib import Path as Filepath
from datetime import datetime
import pandas as pd

script_dir = Filepath(__file__).parent
# 构造目标data目录的绝对路径（向上一级到scripts目录，再创建data子目录）
data_dir = script_dir.parent / "data"  # 等价于：scripts目录下的data子目录

# 创建目录（parents=True自动创建缺失的父目录，exist_ok=True允许目录已存在）
data_dir.mkdir(parents=True, exist_ok=True)
print(f"已创建/确认目录：{data_dir}")