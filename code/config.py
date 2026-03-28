# config.py

# --- 输出根目录 ---
BASE_DATA_DIR = "../data"

# --- 区域设置 ---
REGION_TRAIN = "NY"
REGION_TEST = "NY"
YEAR = "2020"

# --- 划分参数 ---
SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42

# --- 自动推导路径 (通常不需要修改) ---
import os
CENSUS_DIR = os.path.join(BASE_DATA_DIR, f"CCensusTract{YEAR}")
NID_DIR = os.path.join(BASE_DATA_DIR, f"Nid/{REGION_TRAIN}")
LODES_DIR = os.path.join(BASE_DATA_DIR, "LODES")

def ensure_dirs():
    for p in [CENSUS_DIR, NID_DIR, LODES_DIR]:
        os.makedirs(p, exist_ok=True)