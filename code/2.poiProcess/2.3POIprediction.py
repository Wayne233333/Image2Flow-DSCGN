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

    # 2. 加载待预测数据
    feat_csv_path = os.path.join(config.DATA_DIR, "Vis", f"train_on_{config.REGION}_{config.YEAR}.csv")
    feat_df = pd.read_csv(feat_csv_path)
    feat_df['geocode'] = feat_df['geocode'].astype(str)

    # 3. 执行预测（仅使用遥感特征向量）
    X_input = feat_df[feature_cols].values
    X_scaled = scaler_x.transform(X_input)
    
    logging.info(f"正在生成预测数据...")
    y_pred_std = model.predict(X_scaled)
    
    # 4. 逆变换还原真实数值
    y_pred_log = scaler_y.inverse_transform(y_pred_std)
    y_pred_final = np.expm1(y_pred_log) # 还原 e^x - 1

    y_pred_final = np.clip(y_pred_final, 0, None)
    
    result_df = pd.DataFrame(y_pred_final, columns=target_columns)
    result_df.insert(0, 'geocode', feat_df['geocode'].values)
    
    for col in ['amenity', 'shop']:
        if col in result_df.columns:
            result_df[col] = result_df[col].round().astype(int)
    
    poi_csv_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_POI.csv")
    if os.path.exists(poi_csv_path):
        logging.info(f"检测到原始数据，正在增加 _origin 列...")
        poi_df = pd.read_csv(poi_csv_path)
        poi_df['geocode'] = poi_df['geocode'].astype(str)
        
        origin_cols = ['geocode', 'area', 'amenity', 'shop']
        poi_origin = poi_df[origin_cols].rename(columns={
            'area': 'area_origin',
            'amenity': 'amenity_origin',
            'shop': 'shop_origin'
        })
        
        result_df = pd.merge(result_df, poi_origin, on='geocode', how='left')
    
    # 5. 保存结果
    output_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_pred.csv")
    result_df.to_csv(output_path, index=False)
    logging.info(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    predict_and_save_clean_results()