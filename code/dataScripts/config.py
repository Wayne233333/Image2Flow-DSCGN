# config.py

# --- 原始输入 ---
SHP_PATH = "./data/NY/shapefile/tl_2020_36_tract.shp"       # 区域面状SHP文件
OD_FOLDER = "./data/NY/OD/"               # 原始OD CSV文件夹

# --- 输出根目录 ---
BASE_DATA_DIR = "./data"

# --- 区域设置 ---
REGION_NAME = "NY"                               # 对应训练脚本的 --region
YEAR = "2020"                                    # 数据年份

# --- 划分参数 ---
SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42

# --- 自动推导路径 (通常不需要修改) ---
import os
REGION_PREFIX = REGION_NAME.split('t')[0]
CENSUS_DIR = os.path.join(BASE_DATA_DIR, "CensusTract2020")
NID_DIR = os.path.join(BASE_DATA_DIR, f"Nid/{REGION_PREFIX}")
LODES_DIR = os.path.join(BASE_DATA_DIR, "LODES")

def ensure_dirs():
    for p in [CENSUS_DIR, NID_DIR, LODES_DIR]:
        os.makedirs(p, exist_ok=True)