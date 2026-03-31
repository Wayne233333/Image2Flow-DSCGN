import os
import geopandas as gpd
import sys
sys.path.append('..')
import config as config

def sync_shp_to_tiles():
    # --- 配置路径 ---
    # 卫星影像存放目录 (例如 NY/s2_2020/)
    tile_dir = f"{config.DATA_DIR}/{config.REGION}/s2_2020/"
    # 原始 SHP 路径
    input_shp = config.SHP_PATH 
    # 输出同步后的新 SHP 路径
    output_shp = os.path.join(tile_dir, os.path.basename(input_shp).replace(".shp", "_synced.shp"))

    print(f"正在扫描影像文件夹: {tile_dir}...")
    
    # 1. 提取文件夹内所有影像的 geocode
    # 假设文件名格式为: s2_2020_060371011101000.tif
    # 我们取最后 15 位数字作为 geocode
    tile_files = [f for f in os.listdir(tile_dir) if f.endswith('.tif')]
    existing_geocodes = set()
    for f in tile_files:
        # 去掉后缀，按下划线分割取最后一段
        geocode = f.replace('.tif', '').split('_')[-1]
        existing_geocodes.add(geocode)

    print(f"检测到 {len(existing_geocodes)} 个有效影像瓦片。")

    # 2. 读取 SHP 并过滤
    gdf = gpd.read_file(input_shp)
    # 格式化 SHP 里的 GEOID
    gdf['geocode_tmp'] = gdf['GEOID'].astype(str).apply(
        lambda x: '0' + x if len(x) == 10 else x
    )

    initial_count = len(gdf)
    # 只保留那些在影像列表中存在的元素
    gdf_filtered = gdf[gdf['geocode_tmp'].isin(existing_geocodes)].copy()
    
    final_count = len(gdf_filtered)
    print(f"同步完成: 原始元素 {initial_count} -> 剩余元素 {final_count}")
    print(f"删除了 {initial_count - final_count} 个缺少影像的元素。")

    # 3. 保存更新后的 SHP
    gdf_filtered.drop(columns=['geocode_tmp'], inplace=True)
    gdf_filtered.to_csv(output_shp.replace(".shp", ".csv")) # 同时存一份CSV方便调试
    gdf_filtered.to_file(output_shp)
    print(f"同步后的 SHP 已保存至: {output_shp}")

if __name__ == "__main__":
    sync_shp_to_tiles()