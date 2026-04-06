import os
import pandas as pd
import numpy as np
import joblib
import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.multioutput import MultiOutputRegressor

import sys
sys.path.append("..")
import config as config

# 路径配置
feat_csv_path = os.path.join(config.DATA_DIR, "Vis", f"train_on_{config.REGION}_{config.YEAR}.csv")
poi_csv_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_POI.csv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_refined_model():
    logging.info("读取数据并提取高价值特征...")
    feat_df = pd.read_csv(feat_csv_path)
    poi_df = pd.read_csv(poi_csv_path)

    feat_df['geocode'] = feat_df['geocode'].astype(str)
    poi_df['geocode'] = poi_df['geocode'].astype(str)
    
    merged = pd.merge(feat_df, poi_df, on='geocode', how='inner')

    # 1. 仅保留高价值目标
    target_columns = ['amenity', 'shop', 'area']
    
    # 2. 特征工程：将剔除的列转化为“环境上下文”密度特征
    # 既然这些列不好预测，我们就把它们当作输入，来辅助预测剩下的目标
    context_cols = ['building', 'highway', 'railway', 'tourism', 'leisure', 'office', 'industrial']
    for col in context_cols:
        merged[f'{col}_density'] = merged[col] / (merged['area'] + 0.001)

    # 地理分区特征
    merged['county_code'] = merged['geocode'].str[:5]
    county_dummies = pd.get_dummies(merged['county_code'], prefix='county')
    merged = pd.concat([merged, county_dummies], axis=1)

    # 3. 确定特征列
    # 此时 context_cols 变成了输入特征的一部分
    feature_cols = [c for c in merged.columns if c not in target_columns and c not in ['geocode', 'county_code']]
    
    X_raw = merged[feature_cols].values
    y_raw = merged[target_columns].values
    y_log = np.log1p(y_raw)

    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X_raw)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y_log)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 4. 配置更深度的 LGBM
    base_lgbm = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=8,
        min_child_samples=20,
        reg_alpha=0.2,
        reg_lambda=0.2,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model = MultiOutputRegressor(base_lgbm)
    logging.info(f"训练精简版模型，特征数: {X.shape[1]}, 目标: {target_columns}")
    model.fit(X_train, y_train)

    # 5. 评估
    y_pred_std = model.predict(X_test)
    y_pred_log = scaler_y.inverse_transform(y_pred_std)
    y_test_log = scaler_y.inverse_transform(y_test)

    print("\n" + "="*50)
    print(f"{'Target':<12} | {'Accuracy':<10} | {'Precision':<10}")
    print("-" * 50)
    for i, target in enumerate(target_columns):
        threshold = np.median(y_test_log[:, i])
        y_true_bin = (y_test_log[:, i] > threshold).astype(int)
        y_pred_bin = (y_pred_log[:, i] > threshold).astype(int)
        
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        print(f"{target:<12} | {acc:<10.4f} | {prec:<10.4f}")
    print("="*50 + "\n")

    # 6. 保存
    save_path = os.path.join("..", "ckpt", f"{config.REGION}_POI_model.pkl")
    joblib.dump({
        'model': model, 
        'scaler_x': scaler_x, 
        'scaler_y': scaler_y, 
        'feature_cols': feature_cols,
        'target_columns': target_columns
    }, save_path)
    logging.info(f"精简版模型已保存。")

if __name__ == "__main__":
    train_refined_model()