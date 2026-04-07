import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import sys

sys.path.append("..")
import config as config

json_path = os.path.join(config.DATA_DIR, config.REGION, "POI", "export.json")
shp_path = os.path.join(config.DATA_DIR, config.REGION, "shapefile", f"tl_{config.YEAR}_{config.REGION}_tract.shp")
output_csv = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_POI.csv")

def process_and_aggregate_poi():
    target_tags = ['amenity', 'shop']
    
    # 1. 加载普查区 Shapefile
    print("正在加载普查区 Shapefile...")
    census_gdf = gpd.read_file(shp_path)
    
    id_field = 'GEOID' 
    if id_field not in census_gdf.columns:
        possible_ids = ['GEOID20', 'GEOID10', 'AFFGEOID', 'TRACTCE']
        for p_id in possible_ids:
            if p_id in census_gdf.columns:
                id_field = p_id
                break
    print(f"使用字段 '{id_field}' 作为普查区唯一标识。")

    print("正在计算普查区面积...")
    census_gdf['area'] = census_gdf.to_crs(epsg=3857).area / 10**6 

    # 2. 加载并清洗 OSM 数据
    print("正在加载并清洗 OSM 数据...")
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            data = data.get('elements', [])
    except json.JSONDecodeError:
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
            data = json.loads(content)
        else:
            raise ValueError("无法解析 JSON 文件，请检查格式。")

    # 3. 构建 POI 点图层
    poi_list = []
    for item in data:
        if 'lat' in item and 'lon' in item:
            row = {'lat': item['lat'], 'lon': item['lon']}
            tags = item.get('tags', {})
            for tag in target_tags:
                row[tag] = 1 if tag in tags else 0
            poi_list.append(row)
    
    if not poi_list:
        print("警告：未解析出有效 POI。")
        poi_gdf = gpd.GeoDataFrame(columns=['geometry'] + target_tags, crs="EPSG:4326")
    else:
        poi_df = pd.DataFrame(poi_list)
        geometry = [Point(xy) for xy in zip(poi_df['lon'], poi_df['lat'])]
        poi_gdf = gpd.GeoDataFrame(poi_df, geometry=geometry, crs="EPSG:4326")

    # 4. 统一坐标系进行空间关联
    if census_gdf.crs != poi_gdf.crs:
        poi_gdf = poi_gdf.to_crs(census_gdf.crs)

    print("正在进行空间关联...")
    joined = gpd.sjoin(poi_gdf, census_gdf[[id_field, 'geometry']], predicate='within', how='inner')
    
    poi_stats = joined.groupby(id_field)[target_tags].sum().reset_index()

    # 5. 合并回原始普查区列表（保留面积和所有区域）
    print("正在合并所有数据并生成最终 CSV...")
    all_tracts = census_gdf[[id_field, 'area']].copy()
    final_df = pd.merge(all_tracts, poi_stats, on=id_field, how='left')
    
    final_df[target_tags] = final_df[target_tags].fillna(0).astype(int)
    
    final_df.rename(columns={id_field: 'geocode'}, inplace=True)

    keep_cols = ['geocode', 'area', 'amenity', 'shop']
    final_df = final_df[keep_cols]

    print(f"处理完成。总普查区数: {len(final_df)}")
    final_df.to_csv(output_csv, index=False)
    print(f"结果已保存至: {output_csv}")

if __name__ == "__main__":
    process_and_aggregate_poi()