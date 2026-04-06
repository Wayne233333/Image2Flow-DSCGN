import os
import geopandas as gpd
import sys
sys.path.append('..')
import config as config

def sync_shp_to_tiles():

    tile_dir = os.path.join(config.DATA_DIR, config.REGION, config.YEAR)
    input_shp = os.path.join(config.DATA_DIR, config.REGION, "shapefile", f"tl_{config.YEAR}_{config.REGION}_tract.shp")
    output_shp = os.path.join(config.DATA_DIR, config.REGION, "shapefile", f"tl_{config.YEAR}_{config.REGION}_tract_synced.shp")

    print(f"正在扫描影像文件夹: {tile_dir}...")
    
    tile_files = [f for f in os.listdir(tile_dir) if f.endswith('.tif')]
    existing_geocodes = set()
    for f in tile_files:

        geocode = f.replace('.tif', '').split('_')[-1]
        existing_geocodes.add(geocode)

    print(f"检测到 {len(existing_geocodes)} 个有效影像瓦片。")

    gdf = gpd.read_file(input_shp)
    gdf['geocode_tmp'] = gdf['GEOID'].astype(str).apply(
        lambda x: '0' + x if len(x) == 10 else x
    )

    initial_count = len(gdf)
    gdf_filtered = gdf[gdf['geocode_tmp'].isin(existing_geocodes)].copy()
    
    final_count = len(gdf_filtered)
    print(f"同步完成: 原始元素 {initial_count} -> 剩余元素 {final_count}")
    print(f"删除了 {initial_count - final_count} 个缺少影像的元素。")

    gdf_filtered.drop(columns=['geocode_tmp'], inplace=True)
    gdf_filtered.to_csv(output_shp.replace(".shp", ".csv"))
    gdf_filtered.to_file(output_shp)
    print(f"同步后的 SHP 已保存至: {output_shp}")

if __name__ == "__main__":
    sync_shp_to_tiles()