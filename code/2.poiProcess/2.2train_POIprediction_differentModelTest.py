import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

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

    # 2. 特征预处理：log1p 变换（POI计数数据偏斜）
    X_raw = reg_df[available_features].values.copy()
    y_raw = reg_df['total_inflow'].values.copy()
    
    # 对 POI 计数列做 log1p（area 也做，因为面积也是正偏分布）
    X_log = np.log1p(X_raw)
    y_log = np.log1p(y_raw)
    
    # 3. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_log, y_log, test_size=0.2, random_state=42
    )
    logging.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 4. 候选模型
    candidates = {}

    # --- 模型1: 线性回归 + 标准化 ---
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    pipe_lr.fit(X_train, y_train)
    r2_test = r2_score(y_test, pipe_lr.predict(X_test))
    r2_train = r2_score(y_train, pipe_lr.predict(X_train))
    candidates["LinearRegression"] = (pipe_lr, r2_train, r2_test)

    # --- 模型2: Ridge + 标准化 ---
    pipe_ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0))
    ])
    pipe_ridge.fit(X_train, y_train)
    r2_test = r2_score(y_test, pipe_ridge.predict(X_test))
    r2_train = r2_score(y_train, pipe_ridge.predict(X_train))
    candidates["Ridge"] = (pipe_ridge, r2_train, r2_test)

    # --- 模型3: Ridge + 多项式特征(degree=2) ---
    pipe_poly = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge(alpha=10.0))
    ])
    pipe_poly.fit(X_train, y_train)
    r2_test = r2_score(y_test, pipe_poly.predict(X_test))
    r2_train = r2_score(y_train, pipe_poly.predict(X_train))
    candidates["Ridge+Poly2"] = (pipe_poly, r2_train, r2_test)

    # --- 模型4: 随机森林 ---
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    r2_test = r2_score(y_test, rf.predict(X_test))
    r2_train = r2_score(y_train, rf.predict(X_train))
    candidates["RandomForest"] = (rf, r2_train, r2_test)

    # 5. 输出对比结果
    print(f"\n{'='*70}")
    print(f"模型对比 (特征: {available_features})")
    print(f"{'='*70}")
    print(f"{'模型':<20} | {'Train R²':<10} | {'Test R²':<10}")
    print(f"{'-'*70}")
    for name, (_, r2_tr, r2_te) in sorted(candidates.items(), key=lambda x: x[1][2], reverse=True):
        marker = " ★" if name == max(candidates, key=lambda k: candidates[k][2]) else ""
        print(f"{name:<20} | {r2_tr:<10.4f} | {r2_te:<10.4f}{marker}")
    print(f"{'='*70}")

    # 6. 选择测试集 R² 最高的模型
    best_name = max(candidates, key=lambda k: candidates[k][2])
    best_pipe, best_r2_train, best_r2_test = candidates[best_name]
    
    print(f"\n最优模型: {best_name}")
    print(f"Train R²: {best_r2_train:.4f}, Test R²: {best_r2_test:.4f}")

    # 7. 用全量数据重新训练最优模型
    logging.info(f"用全量数据重新训练 {best_name}...")
    
    # 重建同类型模型并 fit 全量数据
    if best_name == "LinearRegression":
        final_model = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    elif best_name == "Ridge":
        final_model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=10.0))])
    elif best_name == "Ridge+Poly2":
        final_model = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=2, include_bias=False)), ('ridge', Ridge(alpha=10.0))])
    elif best_name == "RandomForest":
        final_model = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
    
    final_model.fit(X_log, y_log)
    full_r2 = r2_score(y_log, final_model.predict(X_log))
    print(f"全量数据 R²: {full_r2:.4f}")

    # 8. 保存模型
    save_path = os.path.join("..", "ckpt", f"{config.REGION}_POI_model.pkl")
    joblib.dump({
        'model': final_model,
        'features': available_features,
        'model_name': best_name,
        'test_r2': best_r2_test,
        'full_r2': full_r2
    }, save_path)
    logging.info(f"最优模型已保存至: {save_path}")

if __name__ == "__main__":
    train_regression_model()