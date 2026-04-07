import pandas as pd
import numpy as np
import torch
import os
import logging
import argparse
import dgl
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataset import utils
from modules.gnn import MyModelBlock

import sys
sys.path.append('..')
import config as config

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str, default=config.REGION)
parser.add_argument('--year', type=str, default=config.YEAR)
parser.add_argument('--node_feats_path', type=str, default=os.path.join(config.DATA_DIR, 'Vis', f'train_on_{config.REGION}_{config.YEAR}.csv'))
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--micro_batch_size', type=int, default=64)
parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--embedding_size', type=int, default=512)
parser.add_argument('--evaluate_every', type=int, default=5)

def train(args, logger):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    region = args.region
    
    # 1. 加载数据集
    mappath = os.path.join(config.DATA_DIR, f'CensusTract{args.year}', f'nodeid_geocode_mapping_{region}.csv')
    data = utils.load_nids_dataset(
        node_feats_path=args.node_feats_path, 
        year=args.year, 
        region=region, 
        mappath=mappath
    )
    
    train_nids = data['train_nids']
    valid_nids = data['valid_nids']
    test_nids = data['test_nids']
    node_feats = data['node_feats']
    odflows = data['odflows']
    adj = data['weighted_adjacency']
    num_nodes = data['num_nodes']

    # 2. 构建 DSGNN 基础图
    g = utils.build_graph_from_matrix(adj, node_feats.astype(np.float32), device)
    
    # 【DSGNN 核心】：加载势能对角矩阵 V 并挂载到节点上
    try:
        v_matrix_path = os.path.join('../data', region, 'POI', f'{region}_V_matrix.npy')
        if os.path.exists(v_matrix_path):
            v_matrix = np.load(v_matrix_path)
            v_diag = np.diag(v_matrix).astype(np.float32)
            g.ndata['v'] = torch.from_numpy(v_diag).to(device)
            logger.info(f"成功加载并挂载势能矩阵 V，节点数: {len(v_diag)}")
        else:
            logger.warning(f"未找到势能矩阵文件 {v_matrix_path}，模型将退化为标准 GNN。")
    except Exception as e:
        logger.error(f"加载势能矩阵出错: {e}")

    # 3. 初始化模型与优化器
    # 计算初始噪音 sigma (用于 BMCLoss)
    target_counts = odflows[np.isin(odflows[:, 0], train_nids)][:, 2]
    init_noise_sigma = np.std(np.log2(target_counts + 1.0))
    
    model = MyModelBlock(
        num_nodes=num_nodes, 
        in_dim=node_feats.shape[1], 
        h_dim=args.embedding_size, 
        num_hidden_layers=args.num_hidden_layers,
        init_noise_sigma=init_noise_sigma,
        device=device
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4. 准备数据迭代器 (使用邻域采样防止图过大)
    sampler = dgl.dataloaders.MultiLayerFullNeighborSampler(args.num_hidden_layers)
    train_dataloader = dgl.dataloaders.DataLoader(
        g, train_nids, sampler,
        batch_size=args.micro_batch_size, shuffle=True, drop_last=False
    )
    valid_dataloader = dgl.dataloaders.DataLoader(
        g, valid_nids, sampler,
        batch_size=args.micro_batch_size, shuffle=False, drop_last=False
    )

    # 5. 训练循环
    best_rmse = float('inf')
    accumulation_steps = args.batch_size // args.micro_batch_size
    
    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            # 获取当前 Batch 的 OD 标签
            mask = np.isin(odflows[:, 0], output_nodes.cpu()) & np.isin(odflows[:, 1], g.nodes().cpu())
            if not mask.any(): continue
            
            trip_od = torch.from_numpy(odflows[mask][:, :2].astype(np.int64)).to(device)
            trip_vol = torch.from_numpy(odflows[mask][:, 2].astype(float)).to(device)
            log_trip_vol = utils.log_transform(trip_vol)
            
            # 前向传播与 Loss 计算
            loss = model.get_loss(output_nodes, trip_od, log_trip_vol, blocks)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (it + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()

        # 6. 验证环节
        if epoch % args.evaluate_every == 0 or epoch == args.max_epochs - 1:
            model.eval()
            val_rmse, val_mae, val_cpc = 0, 0, 0
            count = 0
            
            with torch.no_grad():
                for it, (input_nodes, output_nodes, blocks) in enumerate(valid_dataloader):
                    mask_v = np.isin(odflows[:, 0], output_nodes.cpu()) & np.isin(odflows[:, 1], g.nodes().cpu())
                    if not mask_v.any(): continue
                    
                    trip_od_v = torch.from_numpy(odflows[mask_v][:, :2].astype(np.int64)).to(device)
                    trip_vol_v = torch.from_numpy(odflows[mask_v][:, 2].astype(float)).to(device)
                    
                    rmse, mae, cpc = utils.evaluateOne(model, blocks, trip_od_v, trip_vol_v, output_nodes, region=region)
                    val_rmse += rmse; val_mae += mae; val_cpc += cpc; count += 1
            
            avg_rmse = val_rmse / count
            logger.info(f"Epoch {epoch:03d} | Loss: {total_loss:.4f} | Val RMSE: {avg_rmse:.4f} | MAE: {val_mae/count:.4f} | CPC: {val_cpc/count:.4f}")
            
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                torch.save(model.state_dict(), f'../ckpt/{region}_dsgnn_best.pth')
                logger.info(">>> 发现更优模型，已保存。")

if __name__ == "__main__":
    args = parser.parse_args()
    
    log_dir = os.path.dirname(args.log) if '/' in args.log else 'log'
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    train(args, logger)