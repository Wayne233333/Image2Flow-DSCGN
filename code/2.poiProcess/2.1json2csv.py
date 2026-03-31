import json
import pandas as pd
import sys
sys.path.append("..")
import config as config
import os

jsonfile = os.path.join(config.DATA_DIR, config.REGION, "POI", "export.json")
outputfile = os.path.join(config.DATA_DIR, config.REGION, "POI", f"{config.REGION}_POI.csv")

def process_osm_json(input_file, output_file):

    target_tags = [
        'amenity', 'shop', 'building', 'highway', 'railway', 
        'tourism', 'leisure', 'office', 'industrial'
    ]
    all_columns = ['id', 'name', 'lat', 'lon'] + target_tags

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # 移除外层的 { } 大括号（如果存在）
        content = content.strip()
        if content.startswith('{') and content.endswith('}'):
            # 找到第一个 [ 和最后一个 ]
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]
        data = json.loads(content)

    processed_list = []

    # 2. 遍历并展平数据
    for item in data:
        # 基础数据
        row = {
            'id': item.get('id'),
            'lat': item.get('lat'),
            'lon': item.get('lon')
        }
        
        tags = item.get('tags', {})
        
        row['name'] = tags.get('name', '')
        
        for tag_name in target_tags:
            row[tag_name] = tags.get(tag_name, '')

        processed_list.append(row)

    df = pd.DataFrame(processed_list)
    
    df = df[all_columns]
    
    df = df.sort_values(by=['lat', 'lon'], ascending=[True, True])

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"处理完成！文件已保存至: {output_file}")

process_osm_json(jsonfile, outputfile)