# config.py
import os

# --- 输出根目录 ---
DATA_DIR = os.path.join("..", "..", "data")

# --- 区域设置 ---
REGION = "LA"
MODEL = "NY"
YEAR = "2020"

# --- 划分参数 ---
SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42

# --- 自动推导路径 (通常不需要修改) ---
CENSUS_DIR = os.path.join(DATA_DIR, f"CensusTract{YEAR}")
NID_DIR = os.path.join(DATA_DIR, f"Nid/{REGION}")
LODES_DIR = os.path.join(DATA_DIR, "LODES")

def ensure_dirs():
    for p in [CENSUS_DIR, NID_DIR, LODES_DIR]:
        os.makedirs(p, exist_ok=True)