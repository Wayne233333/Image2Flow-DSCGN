import pandas as pd
import numpy as np
import torch
import os
import dgl
import pickle
import warnings

import sys
sys.path.append('..')
import config as config
import os
    
def load_nids_dataset(node_feats_path=os.path.join('..', 'data', 'Vis', f'default_{config.YEAR}.csv'), poi_pred_path=None, year=config.YEAR, fprefix='CommutingFlow_', region=config.REGION, mappath=os.path.join('..', 'data', 'CensusTract2020', 'nodeid_geocode_mapping.csv')):
    
    region_prefix = region.split('t')[0]
    nid_dir = os.path.join('..', 'data', 'Nid', region_prefix)

    train_nids = pd.read_csv(os.path.join(nid_dir, f'train_nids_{region}.csv'), dtype={'geocode': 'string'})
    val_nids = pd.read_csv(os.path.join(nid_dir, f'valid_nids_{region}.csv'), dtype={'geocode': 'string'})
    test_nids = pd.read_csv(os.path.join(nid_dir, f'test_nids_{region}.csv'), dtype={'geocode': 'string'})
    all_nids = pd.read_csv(os.path.join(nid_dir, f'all_nids_{region_prefix}.csv'), dtype={'geocode': 'string'})
    
    mapping_table = pd.read_csv(mappath.replace(".csv", f"_{region_prefix}.csv"), dtype={'geocode': 'string'})

    flow_dir='../data/LODES/'
    odflows_file = f'{flow_dir}{fprefix}{region_prefix}_{year}gt10.csv'
    odflows = pd.read_csv(odflows_file, dtype={'w_geocode': 'string', 'h_geocode': 'string'})
    odflows = geocode_to_nodeid(odflows, mapping_table)    
    
    # 1. 加载影像视觉特征
    node_feats = pd.read_csv(node_feats_path, dtype={'geocode': 'string'})
    node_feats['geocode'] = mapping_table.set_index('geocode').loc[node_feats['geocode']].values
    node_feats = node_feats.rename(columns={'geocode': 'nid'}).set_index('nid').sort_index()
   
    std = node_feats.std()
    std = std.replace(0, 1e-9) 
    node_feats = (node_feats - node_feats.mean()) / std

    # 2. 【新增】加载预测的 POI 势能特征
    if poi_pred_path and os.path.exists(poi_pred_path):
        poi_feats = pd.read_csv(poi_pred_path, dtype={'geocode': 'string'})
        poi_feats['geocode'] = mapping_table.set_index('geocode').loc[poi_feats['geocode']].values
        poi_feats = poi_feats.rename(columns={'geocode': 'nid'}).set_index('nid').sort_index()
        # 对势能矩阵同样进行标准化，避免梯度爆炸
        p_std = poi_feats.std().replace(0, 1e-9)
        poi_feats = (poi_feats - poi_feats.mean()) / p_std
        poi_feats_val = poi_feats.values
    else:
        poi_feats_val = None

    adjpath = os.path.join('..', 'data', 'CensusTract2020', f'adjacency_matrix_bycar_m_{region_prefix}.csv')
    ct_adj = pd.read_csv(adjpath, dtype={'Unnamed: 0': 'string'}).set_index('Unnamed: 0')
    
    ct_inorder = mapping_table.sort_values(by='node_id')['geocode']
    ct_adj = ct_adj.loc[ct_inorder, ct_inorder.astype(str)].fillna(0)
    
    max_val = ct_adj.max().max()
    ct_adj = ct_adj / (max_val if max_val > 0 else 1.0)
    
    mapping_table = mapping_table.set_index('geocode')
    
    data = {
        'train_nids': mapping_table.loc[train_nids['geocode']].values.ravel(),
        'valid_nids': mapping_table.loc[val_nids['geocode']].values.ravel(),
        'test_nids': mapping_table.loc[test_nids['geocode']].values.ravel(),
        'all_nids': mapping_table.loc[all_nids['geocode']].values.ravel(),
        'odflows': odflows[['src', 'dst', 'count', 'dis_m']].values,
        'num_nodes': ct_adj.shape[0],
        'node_feats': node_feats.values,
        'poi_feats': poi_feats_val,  # 导出势能矩阵
        'weighted_adjacency': ct_adj.values
    }
    return data

def geocode_to_nodeid(dataframe, mapping_table):
    df = dataframe.copy()
    mapping = mapping_table.copy()
    mapping.set_index('geocode', inplace=True)
    df['src'] = mapping.loc[df['h_geocode']].values
    df['dst'] = mapping.loc[df['w_geocode']].values
    return df[['src', 'dst', 'count','dis_m']]
    
def nodeid_to_geocode(dataframe, region):
    df = dataframe.copy()
    region_prefix = region.split('t')[0]
    mapping = pd.read_csv(os.path.join('..', 'data', 'CensusTract2020', f'nodeid_geocode_mapping_{region_prefix}.csv')).copy()           
    mapping.set_index('node_id', inplace=True)
    df['h_geocode'] = mapping.loc[df['src']].values
    df['w_geocode'] = mapping.loc[df['dst']].values
    return df[['h_geocode', 'w_geocode', 'count', 'prediction']]

def build_graph_from_matrix(adj_matrix, node_feats, poi_feats=None, device='cpu'):
    dst, src = adj_matrix.nonzero()
    edge_weights = torch.tensor(adj_matrix[adj_matrix.nonzero()]).float().view(-1, 1)
    g = dgl.DGLGraph()
    g = g.to(device)
    g.add_nodes(adj_matrix.shape[0])
    g.add_edges(src, dst, {'d': edge_weights})
    
    g.ndata['attr'] = torch.from_numpy(node_feats).to(device)
    
    if poi_feats is not None:
        g.ndata['pot'] = torch.from_numpy(poi_feats).to(device)
        
    return g

def evaluateOne(model, g, trip_od, trip_volume, output_nodes):
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
        if os.path.exists(os.path.join('outputs', train_on.split('_')[0]))==False:
          os.makedirs(os.path.join('outputs', train_on.split('_')[0]))
        result.to_csv(os.path.join('outputs', train_on.split('_')[0], 
                                   train_on+'_'+region+'_prediction_'+prefix+'.csv'))
    return rmse.item(), mae.item(), cpc.item()

# 【修复 3】安全的对数与指数变换（解决 NaN 核心）
def log_transform(y):
    return torch.log2(y + 1.0) # 防止 log(0) 

def exp_transform(scaled_y):
    output = torch.clamp(scaled_y, min=0)
    return torch.pow(2, output) - 1.0 # 还原

def RMSE(y_hat, y):
    return torch.sqrt(torch.mean((y_hat - y)**2))

def MAE(y_hat, y):
    return torch.mean(torch.abs(y_hat - y))

def CPC(y_hat, y):
    # 分母加极小值防止除以 0
    return 2 * torch.sum(torch.min(y_hat, y)) / (torch.sum(y_hat) + torch.sum(y) + 1e-9)

def CPC_(y, y_hat):
    # 分母加极小值防止除以 0
    return 2 * np.minimum(y_hat, y).sum() / (y_hat.sum() + y.sum() + 1e-9)