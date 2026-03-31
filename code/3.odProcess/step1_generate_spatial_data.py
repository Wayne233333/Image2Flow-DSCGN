# step1_generate_spatial_data.py
import pandas as pd
import geopandas as gpd
from scipy.spatial import distance_matrix
import sys
sys.path.append('..')
import config as config
import os

def format_geocode(code):
    s = str(code).strip()
    return '0' + s if len(s) == 10 else s

def run():
    config.ensure_dirs()
    print(f"正在处理 SHP 并生成空间结构数据...")
    
    gdf = gpd.read_file(config.SHP_PATH)
    gdf['geocode'] = gdf['GEOID'].astype(str).apply(format_geocode)
    
    # 计算米制中点 (EPSG:3857)
    gdf_meter = gdf.to_crs(epsg=3857)
    gdf_unique = gdf_meter.drop_duplicates(subset=['geocode']).sort_values('geocode')
    unique_geocodes = gdf_unique['geocode'].tolist()

    # 1. 生成 Mapping Table
    mapping_df = pd.DataFrame({'node_id': range(len(unique_geocodes)), 'geocode': unique_geocodes})
    mapping_df.to_csv(os.path.join(config.CENSUS_DIR, f"nodeid_geocode_mapping_{config.REGION_PREFIX}.csv"), index=False)

    # 2. 生成邻接矩阵 (直线距离)
    coords = [(geom.x, geom.y) for geom in gdf_unique.centroid]
    dist_mat = distance_matrix(coords, coords)
    adj_df = pd.DataFrame(dist_mat, index=unique_geocodes, columns=unique_geocodes)
    adj_df.to_csv(os.path.join(config.CENSUS_DIR, f"adjacency_matrix_bycar_m_{config.REGION_PREFIX}.csv"))
    
    print(f"空间数据已存至: {config.CENSUS_DIR}")

if __name__ == "__main__":
    run()