import os
from osgeo import gdal
import geopandas as gpd
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument('--input_tif_path', type=str, default=os.path.join(config.BASE_DATA_DIR, config.REGION, config.REGION + "_" + config.YEAR + "_WGS84.tif"))
parser.add_argument('--output_tif_dir', type=str, default=config.BASE_DATA_DIR)
parser.add_argument('--region', type=str, default=config.REGION) 
parser.add_argument('--year', type=str, default=config.YEAR)
parser.add_argument('--shp_path', type=str, default=os.path.join(config.BASE_DATA_DIR, config.REGION, "shapefile", f"tl_{config.YEAR}_{config.REGION}_tract.shp"))

if __name__ == '__main__':
    args = parser.parse_args()

    shps = gpd.read_file(args.shp_path)
    for _, rows in shps.iterrows():
        data_path = os.path.join(args.output_tif_dir, args.region, args.year)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        tif_out_name = args.year+"_"+rows['GEOID']+".tif"
        tif_out = os.path.join(data_path, tif_out_name)

        ds = gdal.Warp(tif_out, args.input_tif_path,
                        format='GTiff', cutlineDSName=rows['geometry'], cropToCutline=True, dstNodata=0)
        del ds

