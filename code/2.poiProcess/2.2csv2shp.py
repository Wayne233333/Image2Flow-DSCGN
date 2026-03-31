import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import sys
sys.path.append('..')
import config as config
import os

poicsv = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_POI.csv")
os.makedirs(os.path.join(poicsv, "..", "shapefile"), exist_ok=True)
poishp = os.path.join(config.DATA_DIR, config.REGION, "POI", "shapefile", f"{config.REGION}_POI.shp")

def csv_to_shapefile(input_csv, output_shp):

    df = pd.read_csv(input_csv)

    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf.to_file(output_shp, driver='ESRI Shapefile', encoding='utf-8')
    
    print(f"转换成功！Shapefile 已保存至: {output_shp}")

# 调用示例
csv_to_shapefile(poicsv, poishp)