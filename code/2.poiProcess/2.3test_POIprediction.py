import os
import pandas as pd
import numpy as np
import joblib
import logging

import sys
sys.path.append("..")
import config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_and_save_clean_results():
    # 1. 加载模型及配套组件
    model_path = os.path.join("..", "ckpt", f"{config.REGION}_POI_model.pkl")
    if not os.path.exists(model_path):
        logging.error("未找到训练好的模型文件。")
        return

    checkpoint = joblib.load(model_path)
    model = checkpoint['model']
    scaler_x = checkpoint['scaler_x']
    scaler_y = checkpoint['scaler_y']
    feature_cols = checkpoint['feature_cols']
    target_columns = checkpoint['target_columns']

    # 2. 加载待预测数据（用于构造特征矩阵）
    feat_csv_path = os.path.join(config.DATA_DIR, "Vis", f"train_on_{config.REGION}_{config.YEAR}.csv")
    poi_csv_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_POI.csv")
    
    feat_df = pd.read_csv(feat_csv_path)
    poi_df = pd.read_csv(poi_csv_path)
    
    feat_df['geocode'] = feat_df['geocode'].astype(str)
    poi_df['geocode'] = poi_df['geocode'].astype(str)
    merged = pd.merge(feat_df, poi_df, on='geocode', how='inner')

    # 3. 构造特征（需与训练时完全一致）
    context_cols = ['building', 'highway', 'railway', 'tourism', 'leisure', 'office', 'industrial']
    for col in context_cols:
        merged[f'{col}_density'] = merged[col] / (merged['area'] + 0.001)
    
    merged['county_code'] = merged['geocode'].str[:5]
    county_dummies = pd.get_dummies(merged['county_code'], prefix='county')
    merged = pd.concat([merged, county_dummies], axis=1)

    # 补齐训练时存在但当前数据缺失的 One-Hot 列
    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = 0

    # 4. 执行预测
    X_input = merged[feature_cols].values
    X_scaled = scaler_x.transform(X_input)
    
    logging.info(f"正在生成预测数据...")
    y_pred_std = model.predict(X_scaled)
    
    # 5. 逆变换还原真实数值
    y_pred_log = scaler_y.inverse_transform(y_pred_std)
    y_pred_final = np.expm1(y_pred_log) # 还原 e^x - 1

    # 6. 【核心修改】仅保存 geocode 和去掉 pred_ 前缀的预测结果
    # 直接使用原始 target_columns 作为列名
    result_df = pd.DataFrame(y_pred_final, columns=target_columns)
    result_df.insert(0, 'geocode', merged['geocode'].values)
    
    # 保存结果，不包含原始真实值列
    output_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_pred.csv")
    result_df.to_csv(output_path, index=False)
    
    logging.info(f"预测结果已净化并保存至: {output_path}")

if __name__ == "__main__":
    predict_and_save_clean_results()