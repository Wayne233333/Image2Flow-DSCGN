import pandas as pd
import numpy as np
import torch
import os
import dgl
import pickle
import warnings

# 忽略 DGL 的旧版本警告
warnings.filterwarnings('ignore', category=UserWarning, module='dgl')

def load_nids_dataset(node_feats_path='../data/Vis/default.csv', year=2020, fprefix='CommutingFlow_', region='default', mappath='../data/CensusTract2020/nodeid_geocode_mapping.csv'):
    
    region_prefix = region.split('t')[0]
    nid_dir = f"../data/Nid/{region_prefix}/"

    # 1. 加载节点划分
    train_nids = pd.read_csv(f'{nid_dir}train_nids_{region}.csv', dtype={'geocode': 'string'})
    val_nids = pd.read_csv(f'{nid_dir}valid_nids_{region}.csv', dtype={'geocode': 'string'})
    test_nids = pd.read_csv(f'{nid_dir}test_nids_{region}.csv', dtype={'geocode': 'string'})
    all_nids = pd.read_csv(f'{nid_dir}all_nids_{region_prefix}.csv', dtype={'geocode': 'string'})
    
    # 2. 加载 ID 映射表
    actual_mappath = mappath.replace(".csv", f"_{region_prefix}.csv") if not os.path.exists(mappath) else mappath
    mapping_table = pd.read_csv(actual_mappath, dtype={'geocode': 'string'})

    # 3. 加载 OD 流量并转换 ID
    flow_dir = '../data/LODES/'
    odflows_file = f'{flow_dir}{fprefix}{region_prefix}_{year}gt10.csv'
    odflows = pd.read_csv(odflows_file, dtype={'w_geocode': 'string', 'h_geocode': 'string'})
    odflows = geocode_to_nodeid(odflows, mapping_table)    
    
    # 4. 加载节点特征并关联势能值 (DSGNN 核心逻辑)
    node_feats_raw = pd.read_csv(node_feats_path, dtype={'geocode': 'string'})
    
    # 【新增】：尝试加载势能文件
    potential_path = os.path.join('../data', region_prefix, "POI", f"{region_prefix}_potential.csv")
    if os.path.exists(potential_path):
        pot_df = pd.read_csv(potential_path, dtype={'geocode': 'string'})
        node_feats_raw = pd.merge(node_feats_raw, pot_df[['geocode', 'potential_v']], on='geocode', how='left')
        node_feats_raw['potential_v'] = node_feats_raw['potential_v'].fillna(0.5)
    else:
        node_feats_raw['potential_v'] = 1.0 # 默认无势能修正

    # 按照 mapping_table 顺序对齐所有节点特征
    full_node_data = pd.merge(mapping_table, node_feats_raw, on='geocode', how='left').fillna(0)
    
    # 提取势能向量并从特征矩阵中剔除 ID 和势能列
    potential_vec = full_node_data['potential_v'].values
    node_feats = full_node_data.drop(columns=['geocode', 'node_id', 'potential_v'], errors='ignore')
   
    # 【修复 1】安全特征归一化
    std = node_feats.std()
    std = std.replace(0, 1e-9) 
    node_feats = (node_feats - node_feats.mean()) / std

    # 5. 加载邻接矩阵
    adjpath = f'../data/CensusTract2020/adjacency_matrix_bycar_m_{region_prefix}.csv'
    ct_adj = pd.read_csv(adjpath, dtype={'Unnamed: 0': 'string'}).set_index('Unnamed: 0')
    
    ct_inorder = mapping_table.sort_values(by='node_id')['geocode']
    ct_adj = ct_adj.loc[ct_inorder, ct_inorder.astype(str)].fillna(0)
    
    # 【修复 2】防止除零
    max_val = ct_adj.max().max()
    ct_adj = ct_adj / (max_val if max_val > 0 else 1.0)
    
    mapping_table_indexed = mapping_table.set_index('geocode')
    
    data = {
        'train_nids': mapping_table_indexed.loc[train_nids['geocode']].values.ravel(),
        'valid_nids': mapping_table_indexed.loc[val_nids['geocode']].values.ravel(),
        'test_nids': mapping_table_indexed.loc[test_nids['geocode']].values.ravel(),
        'all_nids': mapping_table_indexed.loc[all_nids['geocode']].values.ravel(),
        'odflows': odflows[['src', 'dst', 'count', 'dis_m']].values,
        'num_nodes': ct_adj.shape[0],
        'node_feats': node_feats.values,
        'potential_vec': potential_vec, # 传递势能向量
        'weighted_adjacency': ct_adj.values
    }
    return data

def geocode_to_nodeid(dataframe, mapping_table):
    df = dataframe.copy()
    mapping = mapping_table.copy().set_index('geocode')
    df['src'] = mapping.loc[df['h_geocode']].values
    df['dst'] = mapping.loc[df['w_geocode']].values
    return df[['src', 'dst', 'count', 'dis_m']]
    
def nodeid_to_geocode(dataframe, region):
    df = dataframe.copy()
    region_prefix = region.split('t')[0]
    mapping = pd.read_csv(f'../data/CensusTract2020/nodeid_geocode_mapping_{region_prefix}.csv').copy()           
    mapping.set_index('node_id', inplace=True)
    df['h_geocode'] = mapping.loc[df['src']].values
    df['w_geocode'] = mapping.loc[df['dst']].values
    return df[['h_geocode', 'w_geocode', 'count', 'prediction']]

def build_graph_from_matrix(adj_matrix, node_feats, device='cpu', potential_vec=None):
    dst, src = adj_matrix.nonzero()
    edge_weights = torch.tensor(adj_matrix[adj_matrix.nonzero()]).float().view(-1, 1)
    
    # 使用 DGL 最新推荐方式创建图
    g = dgl.graph((src, dst), num_nodes=adj_matrix.shape[0]).to(device)
    g.edata['d'] = edge_weights.to(device)
    g.ndata['attr'] = torch.from_numpy(node_feats.astype(np.float32)).to(device)
    
    # 【DSGNN 关键】：将势能向量挂载为节点数据 v
    if potential_vec is not None:
        g.ndata['v'] = torch.from_numpy(potential_vec.astype(np.float32)).to(device)
        
    return g

def evaluateOne(model, g, trip_od, trip_volume, output_nodes):
    model.eval()
    with torch.no_grad():
        node_embedding = model(g)
        log_prediction = model.predict_edge(node_embedding, trip_od, output_nodes)
        prediction = exp_transform(log_prediction)
        y = trip_volume.float().view(-1, 1)
        rmse = RMSE(prediction, y)
        mae = MAE(prediction, y)
        cpc = CPC(prediction, y)
    return rmse.item(), mae.item(), cpc.item()

def evaluateOutput(model, g, trip_od, trip_volume, output_nodes, region, prefix, train_on):
    model.eval()
    with torch.no_grad():
        node_embedding = model(g)
        log_prediction = model.predict_edge(node_embedding, trip_od, output_nodes)
        prediction = exp_transform(log_prediction)
        y = trip_volume.float().view(-1, 1)

        rmse = RMSE(prediction, y)
        mae = MAE(prediction, y)
        cpc = CPC(prediction, y)
        
        result = pd.DataFrame(torch.cat((trip_od, y, prediction), 1).cpu().numpy(), 
                              columns=['src','dst','count','prediction'])
        result = nodeid_to_geocode(result,region)
        
        save_dir = os.path.join('outputs', train_on.split('_')[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, f"{train_on}_{region}_prediction_{prefix}.csv")
        result.to_csv(save_path, index=False)
        
    return rmse.item(), mae.item(), cpc.item()

# 【修复 3】安全的对数与指数变换
def log_transform(y):
    return torch.log2(y + 1.0) 

def exp_transform(scaled_y):
    # 增加上限 clamp 防止 pow(2, x) 溢出导致 inf
    output = torch.clamp(scaled_y, min=0, max=20) 
    return torch.pow(2, output) - 1.0

def RMSE(y_hat, y):
    return torch.sqrt(torch.mean((y_hat - y)**2))

def MAE(y_hat, y):
    return torch.mean(torch.abs(y_hat - y))

def CPC(y_hat, y):
    return 2 * torch.sum(torch.min(y_hat, y)) / (torch.sum(y_hat) + torch.sum(y) + 1e-9)

def CPC_(y, y_hat):
    # 修正 Numpy 版：np.minimum 需要处理数组
    return 2 * np.minimum(y_hat, y).sum() / (y_hat.sum() + y.sum() + 1e-9)