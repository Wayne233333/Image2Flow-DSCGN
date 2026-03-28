# step3_split_datasets.py
import pandas as pd
from sklearn.model_selection import train_test_split
import config
import os

def run():
    config.ensure_dirs()
    map_path = os.path.join(config.CENSUS_DIR, f"nodeid_geocode_mapping_{config.REGION_PREFIX}.csv")
    unique_geocodes = pd.read_csv(map_path, dtype={'geocode': str})['geocode'].tolist()

    print(f"划分数据集 (种子: {config.RANDOM_SEED})...")
    train, temp = train_test_split(unique_geocodes, test_size=(1 - config.SPLIT_RATIO["train"]), random_state=config.RANDOM_SEED)
    val, test = train_test_split(temp, test_size=(config.SPLIT_RATIO["test"] / (1 - config.SPLIT_RATIO["train"])), random_state=config.RANDOM_SEED)

    # 导出索引文件
    pd.DataFrame({'geocode': train}).to_csv(os.path.join(config.NID_DIR, f"train_nids_{config.REGION_NAME}.csv"), index=False)
    pd.DataFrame({'geocode': val}).to_csv(os.path.join(config.NID_DIR, f"valid_nids_{config.REGION_NAME}.csv"), index=False)
    pd.DataFrame({'geocode': test}).to_csv(os.path.join(config.NID_DIR, f"test_nids_{config.REGION_NAME}.csv"), index=False)
    pd.DataFrame({'geocode': unique_geocodes}).to_csv(os.path.join(config.NID_DIR, f"all_nids_{config.REGION_PREFIX}.csv"), index=False)

    print(f"划分完成。索引文件已存至: {config.NID_DIR}")

if __name__ == "__main__":
    run()