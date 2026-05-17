import os
import pandas as pd
import numpy as np
import joblib
import logging

import sys
sys.path.append("..")
import config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_and_save():
    # 1. 加载回归模型
    model_path = os.path.join("..", "ckpt", f"{config.REGION}_POI_model.pkl")
    if not os.path.exists(model_path):
        logging.error("未找到训练好的模型文件，请先运行 2.2train_POIprediction.py")
        return

    checkpoint = joblib.load(model_path)
    model = checkpoint['model']
    features = checkpoint['features']

    # 2. 读取 POI 真实数据
    poi_csv_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_POI.csv")
    poi_df = pd.read_csv(poi_csv_path)
    poi_df['geocode'] = poi_df['geocode'].astype(str)

    # 3. 特征预处理：log1p 变换（与训练时一致）
    X_raw = poi_df[features].values
    X_log = np.log1p(X_raw)

    # 4. 预测并还原（模型输出是 log1p 空间的值）
    y_pred_log = model.predict(X_log)
    poi_df['gravity_A'] = np.expm1(y_pred_log)  # 还原到原始空间
    poi_df['gravity_A'] = np.clip(poi_df['gravity_A'], 0, None)
    
    logging.info(f"已计算 {len(poi_df)} 个普查区的综合吸引力 (gravity_A)")

    # 5. 输出 pred.csv
    output_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_pred.csv")
    
    output_cols = ['geocode'] + features + ['gravity_A']
    result_df = poi_df[output_cols].copy()
    result_df.to_csv(output_path, index=False)
    logging.info(f"结果已保存至: {output_path}")
    print(f"\n预览 (前5行):")
    print(result_df.head().to_string(index=False))

if __name__ == "__main__":
    predict_and_save()