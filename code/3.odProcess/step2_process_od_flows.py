import pandas as pd
import geopandas as gpd
import os
import glob
from shapely.geometry import Point
import sys
sys.path.append('..')
import config as config

def process_od_data_with_distance(od_folder, shp_path, output_path):
    # 1. 读取 SHP 文件
    print(f"正在读取 SHP 文件: {shp_path}...")
    gdf = gpd.read_file(shp_path)
    
    # 确保 GEOID 为字符串并补齐格式
    gdf['GEOID'] = gdf['GEOID'].astype(str).apply(lambda x: '0' + x if len(x) == 10 else x)
    
    # 2. 计算中点并处理投影以获得精确的米制距离
    # 将坐标系转为米制单位 (EPSG:3857) 以计算距离
    print("正在计算区域中点及距离矩阵...")
    gdf_meter = gdf.to_crs(epsg=3857) 
    gdf_meter['centroid'] = gdf_meter.centroid
    
    # 建立 ID 到 中点坐标的映射字典
    # 结果为 { 'geoid': (x, y) }
    centroid_dict = pd.Series(gdf_meter.centroid.values, index=gdf_meter['GEOID']).to_dict()
    valid_geocodes = set(centroid_dict.keys())

    # 3. 准备处理 OD 数据
    all_csv_files = glob.glob(os.path.join(od_folder, "*.csv"))
    combined_data = []

    def format_geocode(code):
        s = str(code).strip()
        return '0' + s if len(s) == 14 else s

    for file_path in all_csv_files:
        print(f"正在处理: {os.path.basename(file_path)}...")
        chunks = pd.read_csv(
            file_path, 
            usecols=['h_geocode', 'w_geocode', 'S000'],
            dtype={'h_geocode': str, 'w_geocode': str, 'S000': float},
            chunksize=200000
        )

        for chunk in chunks:
            chunk['h_geocode'] = chunk['h_geocode'].apply(format_geocode).str[:11]
            chunk['w_geocode'] = chunk['w_geocode'].apply(format_geocode).str[:11]

            # 过滤起讫点
            filtered_chunk = chunk[
                chunk['h_geocode'].isin(valid_geocodes) & 
                chunk['w_geocode'].isin(valid_geocodes)
            ].copy()
            
            if not filtered_chunk.empty:
                combined_data.append(filtered_chunk)

    if not combined_data:
        print("未找到有效数据。")
        return

    # 4. 合并并计算距离
    final_df = pd.concat(combined_data, ignore_index=True)
    final_df = final_df.groupby(['h_geocode', 'w_geocode'], as_index=False)['S000'].sum()
    final_df.columns = ['h_geocode', 'w_geocode', 'count']

    # 过滤count小于10的数据
    final_df = final_df[final_df['count'] >= 10]

    print("正在计算 OD 对之间的距离 (dis_m)...")
    
    # 定义距离计算函数
    def calculate_dist(row):
        p1 = centroid_dict[row['h_geocode']]
        p2 = centroid_dict[row['w_geocode']]
        return p1.distance(p2) # 在 3857 投影下单位为米

    # 应用计算
    final_df['dis_m'] = final_df.apply(calculate_dist, axis=1)

    # 转换为整数
    final_df['count'] = final_df['count'].astype(int)
    final_df['dis_m'] = final_df['dis_m'].astype(int)

    # 5. 导出
    final_df.to_csv(output_path, index=False)
    print(f"处理完成！最终文件包含 count 和 dis_m 字段。")
    print(f"保存路径: {output_path}")

if __name__ == "__main__":
    OD_FOLDER = config.OD_FOLDER
    SHP_PATH = config.SHP_PATH
    OUTPUT_FILE = os.path.join(config.LODES_DIR, f"CommutingFlow_{config.REGION_NAME}_{config.YEAR}gt10.csv")

    process_od_data_with_distance(OD_FOLDER, SHP_PATH, OUTPUT_FILE)