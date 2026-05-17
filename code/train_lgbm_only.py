import pandas as pd
import numpy as np
import torch
import os
import pickle
import dgl
import lightgbm as lgb
from dataset import utils
from modules.gnn import MyModelBlock

# --- 参数配置 ---
CONFIG = {
    'log_name': 'OD_LA', 
    'node_feats_path': '../data/Vis/train_on_LA_2020.csv', 
    'region': 'LA',
    'year': '2020',
    'device': 'cuda:0',
    'num_hidden_layers': 2,
    'embedding_size': 512,
    'batch_size': 512,  # 推理时的批大小，若还报错可调至 128
}

def extract_embeddings(model, g, nids, odflows, device):
    model.eval()
    # 找出所有涉及到的源节点和目的节点
    involved_nodes = np.unique(np.append(nids, odflows[np.isin(odflows[:, 0], nids)][:, 1])).astype('int64')

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(CONFIG['num_hidden_layers'] + 1)
    # sampler = dgl.dataloading.MultiLayerNeighborSampler([200] * (CONFIG['num_hidden_layers'] + 1))

    # 核心修改 2: 分批次加载，避免 batch_size 过大
    dataloader = dgl.dataloading.DataLoader(
        g, torch.from_numpy(involved_nodes).to(device), sampler,
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        drop_last=False
    )
    
    node_embed_map = {}
    print(f"Starting mini-batch inference for {len(involved_nodes)} nodes...")
    
    with torch.no_grad():
        for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            blocks = [b.to(device) for b in blocks]
            # 计算 Embedding
            embeds = model(blocks).cpu().numpy()
            
            # 将结果存入字典
            output_nodes_np = output_nodes.cpu().numpy()
            for i, nid in enumerate(output_nodes_np):
                node_embed_map[int(nid)] = embeds[i]
            
            # 释放显存
            del blocks, embeds
            if it % 20 == 0:
                torch.cuda.empty_cache()

    # 构建 LGBM 训练特征
    print("Constructing features for LGBM...")
    condition = np.isin(odflows[:, 0], nids)
    valid_od = odflows[condition]
    
    X, y = [], []
    for row in valid_od:
        src, dst, count, dist = int(row[0]), int(row[1]), row[2], row[3]
        if src in node_embed_map and dst in node_embed_map:
            # 拼接源节点嵌入、目的节点嵌入和距离
            feat = np.concatenate([node_embed_map[src], node_embed_map[dst], [dist]])
            X.append(feat)
            y.append(count)
            
    return np.array(X), np.array(y)

def main():
    device = torch.device(CONFIG['device'])
    torch.cuda.empty_cache() # 运行前清理显存
    
    # 1. 加载数据
    print(f"Loading {CONFIG['region']} dataset...")
    data = utils.load_nids_dataset(
        node_feats_path=CONFIG['node_feats_path'], 
        region=CONFIG['region'], 
        year=CONFIG['year']
    )
    
    # 2. 初始化模型并加载权重
    print("Loading GNN checkpoint and aligning weights...")
    # 注意：针对跨城市预测，这里保持 data['num_nodes']
    model = MyModelBlock(
        data['num_nodes'], 
        in_dim=data['node_feats'].shape[1], 
        h_dim=CONFIG['embedding_size'], 
        num_hidden_layers=CONFIG['num_hidden_layers'], 
        device=device
    )
    
    pth_path = f"./ckpt/{CONFIG['log_name']}_layers{CONFIG['num_hidden_layers']}_emb{CONFIG['embedding_size']}.pth"
    if not os.path.exists(pth_path):
        print(f"Error: {pth_path} not found.")
        return

    checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
    
    # 鲁棒加载逻辑
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        curr_state = model.state_dict()
        new_state = {k: v for k, v in checkpoint['state_dict'].items() 
                     if k in curr_state and v.size() == curr_state[k].size()}
        curr_state.update(new_state)
        model.load_state_dict(curr_state)
        print("Used partial weight loading.")

    model.to(device)
    
    # 构建 DGL 图（传入 potential 用于 DSGNN）
    g = utils.build_graph_from_matrix(data['weighted_adjacency'], data['node_feats'].astype(np.float32), data['potential'], device)

    # 3. 提取 Embedding
    print("Extracting embeddings (Memory-safe mode)...")
    X_train, y_train = extract_embeddings(model, g, data['train_nids'], data['odflows'], device)
    
    if len(X_train) == 0:
        print("Error: No training features extracted. Check your data or node IDs.")
        return

    # 4. 训练 LGBM
    print(f"Training LGBM on {len(X_train)} samples...")
    gbm = lgb.LGBMRegressor(max_depth=10, n_estimators=100, seed=42, n_jobs=-1)
    gbm.fit(X_train, y_train)

    # 5. 保存模型
    if not os.path.exists('./models'):
        os.makedirs('./models')

    save_path = f"./models/lgbm_{CONFIG['log_name']}.txt"
    with open(save_path, 'wb') as f:
        pickle.dump(gbm, f)
    
    print(f"Successfully saved: {save_path}")
    print("You can now proceed to test_ODPrediction_and_compare.py")

if __name__ == "__main__":
    main()