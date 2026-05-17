import os
import pandas as pd
import numpy as np

import sys
sys.path.append("..")
import config as config

def build_potential_matrix():
    # 1. 加载 pred.csv（已包含 gravity_A）
    poi_pred_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_pred.csv")
    poi_df = pd.read_csv(poi_pred_path)
    poi_df['geocode'] = poi_df['geocode'].astype(str)
    
    print(f"加载 {len(poi_df)} 个普查区数据")
    print(f"gravity_A 统计: min={poi_df['gravity_A'].min():.2f}, max={poi_df['gravity_A'].max():.2f}, mean={poi_df['gravity_A'].mean():.2f}")

    # 2. 极性反转与归一化计算势能 v_i
    V_max = 1.0
    delta_V = 1.0
    
    A_min = poi_df['gravity_A'].min()
    A_max = poi_df['gravity_A'].max()
    
    if A_max - A_min != 0:
        norm_A = (poi_df['gravity_A'] - A_min) / (A_max - A_min)
        poi_df['potential_v'] = V_max - (norm_A * delta_V)
    else:
        poi_df['potential_v'] = V_max
        
    # 3. 构建对角矩阵 V
    v_vector = poi_df['potential_v'].values
    potential_matrix_V = np.diag(v_vector)
    
    print(f"势能对角矩阵构建完成，维度: {potential_matrix_V.shape}")
    print(f"potential_v 统计: min={poi_df['potential_v'].min():.4f}, max={poi_df['potential_v'].max():.4f}")
    
    # 4. 保存结果供 GNN 使用
    output_v_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_potential.csv")
    poi_df[['geocode', 'potential_v']].to_csv(output_v_path, index=False)
    
    # 保存对角矩阵为 npy 格式
    matrix_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_V_matrix.npy")
    np.save(matrix_path, potential_matrix_V)
    
    print(f"势能数据已保存至: {output_v_path}")
    print(f"矩阵已保存至: {matrix_path}")
    
    return potential_matrix_V

if __name__ == "__main__":
    V = build_potential_matrix()