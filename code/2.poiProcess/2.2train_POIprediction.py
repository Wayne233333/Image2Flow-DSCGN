import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import sys
sys.path.append("..")
import config as config

poi_csv_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_POI.csv")
od_flow_path = os.path.join(config.DATA_DIR, "LODES", f"CommutingFlow_{config.REGION}_{config.YEAR}gt10.csv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALL_FEATURES = ['area', 'amenity', 'shop', 'highway', 'tourism', 'leisure', 'office']

def train_regression_model():
    logging.info("读取数据...")
    poi_df = pd.read_csv(poi_csv_path)
    od_df = pd.read_csv(od_flow_path)
    
    poi_df['geocode'] = poi_df['geocode'].astype(str)
    od_df['w_geocode'] = od_df['w_geocode'].astype(str)

    # 1. 计算每个普查区的流入量
    in_flow_stats = od_df.groupby('w_geocode')['count'].sum().reset_index()
    in_flow_stats.columns = ['geocode', 'total_inflow']
    
    reg_df = pd.merge(in_flow_stats, poi_df, on='geocode', how='inner')
    logging.info(f"匹配到的普查区数量: {len(reg_df)}")

    available_features = [f for f in ALL_FEATURES if f in reg_df.columns]
    logging.info(f"可用特征: {available_features}")

    # 2. 特征预处理：log1p 变换
    X_raw = reg_df[available_features].values.copy()
    y_raw = reg_df['total_inflow'].values.copy()
    X_log = np.log1p(X_raw)
    y_log = np.log1p(y_raw)
    
    # 3. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_log, y_log, test_size=0.2, random_state=42
    )
    logging.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 4. RandomForest 模型
    model = RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    
    print(f"\n{'='*50}")
    print(f"RandomForest 模型评估")
    print(f"{'='*50}")
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test  R²: {r2_test:.4f}")
    print(f"{'='*50}")

    # 5. 用全量数据重新训练
    logging.info("用全量数据重新训练...")
    final_model = RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    final_model.fit(X_log, y_log)
    full_r2 = r2_score(y_log, final_model.predict(X_log))
    print(f"全量数据 R²: {full_r2:.4f}")

    # 6. 保存模型
    save_path = os.path.join("..", "ckpt", f"{config.REGION}_POI_model.pkl")
    joblib.dump({
        'model': final_model,
        'features': available_features,
        'test_r2': r2_test,
        'full_r2': full_r2
    }, save_path)
    logging.info(f"模型已保存至: {save_path}")

if __name__ == "__main__":
    train_regression_model()