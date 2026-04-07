import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append("..")
import config as config

def build_potential_matrix():
    # 1. 加载数据
    poi_pred_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_pred.csv")
    od_flow_path = os.path.join(config.DATA_DIR, "LODES", f"CommutingFlow_{config.REGION}_{config.YEAR}gt10.csv")
    
    poi_df = pd.read_csv(poi_pred_path)
    od_df = pd.read_csv(od_flow_path)
    
    poi_df['geocode'] = poi_df['geocode'].astype(str)
    od_df['h_geocode'] = od_df['h_geocode'].astype(str)
    od_df['w_geocode'] = od_df['w_geocode'].astype(str)

    # 2. 演算不同 POI 类别的权重 (w)
    in_flow_stats = od_df.groupby('w_geocode')['count'].sum().reset_index()
    in_flow_stats.columns = ['geocode', 'total_inflow']
    
    reg_df = pd.merge(in_flow_stats, poi_df, on='geocode', how='inner')
    
    features = ['area', 'amenity', 'shop']
    X = reg_df[features].values
    y = reg_df['total_inflow'].values

    model = LinearRegression()
    model.fit(X, y)
    weights = model.coef_
    intercept = model.intercept_
    
    print(f"计算得到的权重向量 w: {dict(zip(features, weights))}")
    print(f"偏置 b: {intercept:.4f}")

    # 3. 计算综合引力系数 A_i
    poi_values = poi_df[features].values
    gravity_A = np.dot(poi_values, weights) + intercept
    poi_df['gravity_A'] = gravity_A

    # 4. 极性反转与归一化计算势能 v_i
    V_max = 1.0
    delta_V = 1.0
    
    A_min = poi_df['gravity_A'].min()
    A_max = poi_df['gravity_A'].max()
    
    if A_max - A_min != 0:
        norm_A = (poi_df['gravity_A'] - A_min) / (A_max - A_min)
        poi_df['potential_v'] = V_max - (norm_A * delta_V)
    else:
        poi_df['potential_v'] = V_max
        
    # 5. 构建对角矩阵 V
    v_vector = poi_df['potential_v'].values
    potential_matrix_V = np.diag(v_vector)
    
    print(f"势能对角矩阵构建完成，维度: {potential_matrix_V.shape}")
    
    # 6. 保存结果供 GNN 使用
    output_v_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_potential.csv")
    poi_df[['geocode', 'potential_v']].to_csv(output_v_path, index=False)
    
    # 保存对角矩阵为 npy 格式
    matrix_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_V_matrix.npy")
    np.save(matrix_path, potential_matrix_V)
    
    return potential_matrix_V, weights

if __name__ == "__main__":
    V, w = build_potential_matrix()