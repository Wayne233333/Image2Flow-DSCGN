import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import sys

sys.path.append("..")
import config as config

csv_path = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_POI.csv")
shp_path = os.path.join(config.DATA_DIR, config.REGION, "shapefile", f"tl_{config.YEAR}_{config.REGION}_tract.shp")
os.makedirs(os.path.join(config.DATA_DIR, config.REGION, "shapefile_poi"), exist_ok=True)
output_shp_path = os.path.join(config.DATA_DIR, config.REGION, "shapefile_poi", f"tl_{config.YEAR}_{config.REGION}_tract_poi.shp")

expected_columns = ['amenity', 'shop', 'building', 'highway', 'railway', 'tourism', 'leisure', 'office', 'industrial']

print("正在读取数据...")
poi_df = pd.read_csv(csv_path)
census_gdf = gpd.read_file(shp_path)

raw_columns = [str(col).strip() for col in poi_df.columns]
col_mapping = {col.lower(): col for col in raw_columns}

for col in expected_columns:
    census_gdf[col] = 0

print("正在处理 POI 位置信息...")
poi_df = poi_df.dropna(subset=['lat', 'lon'])

geometry = [Point(xy) for xy in zip(poi_df['lon'], poi_df['lat'])]

poi_gdf = gpd.GeoDataFrame(poi_df, geometry=geometry, crs="EPSG:4326")
if census_gdf.crs != poi_gdf.crs:
    print(f"坐标系不一致，正在将 POI 坐标系转换为: {census_gdf.crs}")
    poi_gdf = poi_gdf.to_crs(census_gdf.crs)

print("正在进行空间匹配（点在多边形内判定）...")
joined_gdf = gpd.sjoin(poi_gdf, census_gdf, predicate='within', how='inner')

print("正在统计各普查区的 POI 数量...")
for col in expected_columns:
    if col in col_mapping:
        real_col_name = col_mapping[col]
        
        valid_indices = poi_df[real_col_name].astype(str).str.strip()
        valid_indices = valid_indices[(valid_indices != 'nan') & (valid_indices != '')].index
        
        valid_rows = joined_gdf[joined_gdf.index.isin(valid_indices)]
        
        if len(valid_rows) > 0:
            counts = valid_rows.groupby('index_right').size()
            census_gdf.loc[counts.index, col] = counts
            print(f"字段 [{col}] 统计到了 {counts.sum()} 个有效 POI")
        else:
            print(f"字段 [{col}] 无有效内容")
    else:
        print(f"CSV 中不存在列名: {col}")

print(f"正在导出新的 SHP 文件到: {output_shp_path}")
census_gdf.to_file(output_shp_path, encoding='utf-8')
print("处理完成！")