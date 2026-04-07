import os
import pandas as pd
import numpy as np
import joblib
import logging
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.multioutput import MultiOutputRegressor

import sys
sys.path.append("..")
import config as config

feat_csv_path = os.path.join(config.DATA_DIR, "Vis", f"train_on_{config.REGION}_{config.YEAR}.csv")
poi_csv_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_POI.csv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_metrics(name, y_test_log, y_pred_log, target_columns, scaler_y):
    print(f"\n模型评估: {name}")
    print("="*50)
    print(f"{'Target':<12} | {'Accuracy':<10} | {'Precision':<10}")
    print("-" * 50)
    for i, target in enumerate(target_columns):
        threshold = np.median(y_test_log[:, i])
        y_true_bin = (y_test_log[:, i] > threshold).astype(int)
        y_pred_bin = (y_pred_log[:, i] > threshold).astype(int)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        print(f"{target:<12} | {acc:<10.4f} | {prec:<10.4f}")
    print("="*50)

def train_refined_model():
    logging.info("读取数据...")
    feat_df = pd.read_csv(feat_csv_path)
    poi_df = pd.read_csv(poi_csv_path)

    feat_df['geocode'] = feat_df['geocode'].astype(str)
    poi_df['geocode'] = poi_df['geocode'].astype(str)
    merged = pd.merge(feat_df, poi_df, on='geocode', how='inner')

    target_columns = ['area', 'amenity', 'shop']
    feature_cols = [c for c in feat_df.columns if c not in ['geocode']]
    
    X_raw = merged[feature_cols].values
    y_raw = merged[target_columns].values
    y_log = np.log1p(y_raw)

    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X_raw)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y_log)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    y_test_orig = scaler_y.inverse_transform(y_test)

    # Random Forest
    rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model = MultiOutputRegressor(rf_base)
    logging.info("正在训练随机森林...")
    rf_model.fit(X_train, y_train)
    rf_pred = scaler_y.inverse_transform(rf_model.predict(X_test))
    evaluate_metrics("Random Forest", y_test_orig, rf_pred, target_columns, scaler_y)

    # LightGBM
    lgb_base = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.02, max_depth=8, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model = MultiOutputRegressor(lgb_base)
    logging.info("正在训练 LightGBM...")
    lgb_model.fit(X_train, y_train)
    lgb_pred = scaler_y.inverse_transform(lgb_model.predict(X_test))
    evaluate_metrics("LightGBM", y_test_orig, lgb_pred, target_columns, scaler_y)

    save_path = os.path.join("..", "ckpt", f"{config.REGION}_POI_model.pkl")
    joblib.dump({
        'model': lgb_model, 
        'scaler_x': scaler_x, 
        'scaler_y': scaler_y, 
        'feature_cols': feature_cols,
        'target_columns': target_columns
    }, save_path)
    logging.info(f"模型已保存至: {save_path}")

if __name__ == "__main__":
    train_refined_model()