import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging 
from dataset import utils
from modules.gnn import MyModelBlock
import dgl
import lightgbm as lgb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='log/default.log') 
parser.add_argument('--node_feats_path', type=str, default='../data/Vis/train_on_NY.csv') 
parser.add_argument('--region', type=str, default='NY') 
parser.add_argument('--year', type=str, default='2020') 
parser.add_argument('--device', type=str, default = 'cuda:0')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default = 1e-5)
parser.add_argument('--grad_norm', type=float, default=1.0)
parser.add_argument('--evaluate_every', type=int, default=5)
parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--embedding_size', type=int, default=512)

# 新增参数：物理 Batch Size 与目标累加 Batch Size
parser.add_argument('--micro_batch_size', type=int, default=2, help="防止OOM的物理内存batch size")
parser.add_argument('--target_batch_size', type=int, default=512, help="原版期望的逻辑batch size")

def adjust_learning_rate_warmup(optimizer, epoch, warmup_epochs, initial_lr, max_lr):
    if epoch < warmup_epochs:
        lr = initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            if 'name' in param_group and param_group['name'] == 'noise_sigma':
                continue
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            if 'name' in param_group and param_group['name'] == 'noise_sigma':
                continue
            param_group['lr'] = max_lr

def train(train_args, logger):
    device = torch.device(train_args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    region = train_args.region
    data = utils.load_nids_dataset(year=train_args.year,node_feats_path=train_args.node_feats_path, region=region)

    train_nids = data['train_nids']
    valid_nids = data['valid_nids']
    test_nids = data['test_nids']
    odflows = data['odflows']

    train_nids_d = np.unique(odflows[np.isin(odflows[:, 0], train_nids)][:,1])
    train_nids_od =  np.unique(np.append(train_nids, train_nids_d, axis=0))
    valid_nids_d = np.unique(odflows[np.isin(odflows[:, 0], valid_nids)][:,1])
    valid_nids_od =  np.unique(np.append(valid_nids, valid_nids_d, axis=0))
    test_nids_d = np.unique(odflows[np.isin(odflows[:, 0], test_nids)][:,1])
    test_nids_od =  np.unique(np.append(test_nids, test_nids_d, axis=0))

    node_feats = data['node_feats']
    
    # 【修复 4】安全的噪声初始化
    target_counts = odflows[np.isin(odflows[:, 0], train_nids)][:, 2]
    init_noise_sigma = np.std(np.log10(target_counts + 1.0)) 
    print("init_noise_sigma ", init_noise_sigma)

    ct_adj = data['weighted_adjacency']
    num_nodes = data['num_nodes']

    model = MyModelBlock(num_nodes, in_dim = node_feats.shape[1], h_dim = train_args.embedding_size, num_hidden_layers=train_args.num_hidden_layers,init_noise_sigma = init_noise_sigma, device=device)
    g = utils.build_graph_from_matrix(ct_adj, node_feats.astype(np.float32), device)  
    model.to(device)

    # 设定梯度累加参数
    micro_bs = train_args.micro_batch_size
    accumulation_steps = max(1, train_args.target_batch_size // micro_bs)
    logger.info(f"启用梯度累加: 物理Batch={micro_bs}, 累加步数={accumulation_steps}, 逻辑Batch={micro_bs*accumulation_steps}")

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(train_args.num_hidden_layers+1)
    train_dataloader = dgl.dataloading.DataLoader(
        g, torch.from_numpy(train_nids_od).to(device), sampler,
        batch_size=micro_bs, shuffle=True, drop_last=True)
    valid_dataloader = dgl.dataloading.DataLoader(
        g, torch.from_numpy(valid_nids_od).to(device), sampler,
        batch_size=micro_bs, shuffle=True, drop_last=True)
    test_dataloader = dgl.dataloading.DataLoader(
        g, torch.from_numpy(test_nids_od).to(device), sampler,
        batch_size=micro_bs, shuffle=True, drop_last=True)
    
    model_state_file = './ckpt/{}_layers{}_emb{}.pth'.format(train_args.log.strip('log/').strip('.log'),train_args.num_hidden_layers, train_args.embedding_size)
    best_rmse = float('inf')

    initial_lr = 1e-6
    warmup_epochs = 5
    early_stopping_patience = 20
    best_val_loss = float('inf')
    early_stopping_counter = 0
    noise_var = AverageMeter()
    
    criterion_params = list(model.criterion.parameters())
    criterion_param_ids = list(map(id, criterion_params))
    model_params = filter(lambda p: id(p) not in criterion_param_ids, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': model_params, 'lr': train_args.lr},
        {'params': criterion_params, 'lr': 1e-2}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1)
    
    for epoch in range(train_args.max_epochs):
        model.train()
        adjust_learning_rate_warmup(optimizer, epoch, warmup_epochs, initial_lr, train_args.lr)
        
        # 在 Epoch 开始前清零梯度
        optimizer.zero_grad() 
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            origin_train_nids = output_nodes.cpu()[np.isin(output_nodes.cpu(), train_nids)]
            condition1 = np.isin(odflows[:, 0], origin_train_nids)
            condition2 = np.isin(odflows[:, 1], output_nodes.cpu())
            combined_condition = condition1 & condition2

            trip_od = torch.from_numpy(odflows[combined_condition][:, :2].astype(np.int64)).to(device)

            # 【修复 5】将 [:, -1] 修正为 [:, 2] 以获取真实的 count 标签
            log_trip_volume = utils.log_transform(torch.from_numpy(odflows[combined_condition][:, 2].astype(float))).to(device)
            
            loss = model.get_loss(output_nodes, trip_od, log_trip_volume, blocks)
            
            # 【累加核心逻辑】缩放 Loss 以保证梯度量级与大 Batch 相同
            loss = loss / accumulation_steps
            loss.backward()
            
            noise_var.update(model.criterion.noise_sigma.item() ** 2)

            # 达到指定的累加步数，或者到了当前 Epoch 的最后一步时才更新模型
            if (it + 1) % accumulation_steps == 0 or (it + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_norm)
                optimizer.step()
                optimizer.zero_grad() # 更新后立刻清零
            
        scheduler.step()
        
        if logger.level == logging.DEBUG:
            model.eval()
            # 调试日志仅需前向传播，不涉及反向传播
            with torch.no_grad():
                for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                    origin_train_nids = output_nodes.cpu()[np.isin(output_nodes.cpu(), train_nids)]
                    condition1 = np.isin(odflows[:, 0], origin_train_nids)
                    condition2 = np.isin(odflows[:, 1], output_nodes.cpu())
                    combined_condition = condition1 & condition2

                    trip_od = torch.from_numpy(odflows[combined_condition][:, :2].astype(np.int64)).to(device)
                    trip_volume = torch.from_numpy(odflows[combined_condition][:, 2].astype(float)).to(device)
                    log_trip_volume = utils.log_transform(trip_volume)
                    
                    loss = model.get_loss(output_nodes, trip_od, log_trip_volume, blocks)
                    rmse, mae, cpc = utils.evaluateOne(model, blocks, trip_od, trip_volume, output_nodes)
                    break # 仅打印一个 mini-batch 防止刷屏
                
            logger.debug(f'Epoch: {epoch:04d} - Train Sample - Loss: {loss:.4f} | RMSE: {rmse:.4f} - MAE: {mae:.4f} - CPC: {cpc:.4f}')

        # Validation 部分... (此处仅展示验证集标签修正，其余逻辑相同)
        val_loss = 0
        if epoch % train_args.evaluate_every == 0 or epoch == (train_args.max_epochs-1):
            model.eval()
            for it, (input_nodes, output_nodes, blocks) in enumerate(valid_dataloader):
                with torch.no_grad():
                    # 获取当前 batch 覆盖的起始节点
                    mask = np.isin(output_nodes.cpu(), valid_nids)
                    origin_valid_nids = output_nodes.cpu()[mask]
                    
                    # 筛选 OD 对
                    condition1 = np.isin(odflows[:, 0], origin_valid_nids)
                    condition2 = np.isin(odflows[:, 1], output_nodes.cpu())
                    combined_condition = condition1 & condition2

                    # 【关键修复：防御性检查】
                    if not combined_condition.any():
                        # logger.debug("当前 Batch 无有效 OD 对，跳过...")
                        continue 

                    # 转换为 Tensor
                    trip_od_valid = torch.from_numpy(odflows[combined_condition][:, :2].astype(np.int64)).to(device)
                    trip_volume_valid = torch.from_numpy(odflows[combined_condition][:, 2].astype(float)).to(device)
                    log_trip_volume_valid = utils.log_transform(trip_volume_valid)
                    
                    # 将 blocks 移至设备（防止显存报错）
                    blocks = [b.to(device) for b in blocks]

                    # 计算 Loss
                    loss = model.get_loss(output_nodes, trip_od_valid, log_trip_volume_valid, blocks)
                    
                    # 评估指标
                    rmse, mae, cpc = utils.evaluateOne(model, blocks, trip_od_valid, trip_volume_valid, output_nodes)
        
                logger.info("-----------------------------------------")
                logger.info(f'Epoch: {epoch:04d} - Validation - Loss: {loss:.4f} | RMSE: {rmse:.4f} - MAE: {mae:.4f} - CPC: {cpc:.4f}')
                if rmse < best_rmse:
                    best_rmse = rmse
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'rmse': rmse, 'mae': mae, 'cpc': cpc}, model_state_file)
                    logger.info('Best RMSE found on epoch {}'.format(epoch))
                val_loss = loss
                       
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0  
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    logger.info("Early stopping at epoch {}.".format(epoch))
                    break
            logger.info("-----------------------------------------")


    
    # Test:
    log_base_name = os.path.basename(train_args.log).replace(".log", "")
    prefix = f"{log_base_name}_layers{train_args.num_hidden_layers}_emb{train_args.embedding_size}"
    train_on = log_base_name.replace("bands3", "").replace("bands6", "")
    logger.info(f"----------------------------------------- {region} 0.2 Test")
    
    # Load the best model
    checkpoint = torch.load(model_state_file)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with RMSE {checkpoint['rmse']:.4f}")
    
    model.eval()
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            test_dataloader
    ):
        with torch.no_grad():
            origin_test_nids = output_nodes.cpu()[np.isin(output_nodes.cpu(), test_nids)]
            condition1 = np.isin(odflows[:, 0], origin_test_nids)
            condition2 = np.isin(odflows[:, 1], output_nodes.cpu())
            combined_condition = condition1 & condition2
            trip_od_test = torch.from_numpy(odflows[combined_condition][:, :2].astype(np.int64)).to(device)
            trip_volume_test = torch.from_numpy(odflows[combined_condition][:, -1].astype(float)).to(device)
            log_trip_volume_test = utils.log_transform(trip_volume_test)

            loss = model.get_loss(output_nodes, trip_od_test, log_trip_volume_test, blocks)
        rmse, mae, cpc= utils.evaluateOutput(model, blocks, trip_od_test, trip_volume_test, output_nodes, region, prefix, train_on+"_0.2")
        
        # report
        logger.info("-----------------------------------------")
        logger.info(f'Epoch: {epoch:04d} - Test - Loss: {loss:.4f} | '
                 f'RMSE: {rmse:.4f} - MAE: {mae:.4f} - '
                 f'CPC: {cpc:.4f}')
        
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def extract_embeddings(model, g, nids, odflows, train_args, device):
    condition = np.isin(odflows[:, 0], nids)
    nids_d = np.unique(odflows[condition][:,1]).astype('int64')
    nids_od =  np.unique(np.append(nids, nids_d, axis=0))
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(train_args.num_hidden_layers+1)
    dataloader = dgl.dataloading.DataLoader(
        g, torch.from_numpy(nids_od).to(device), sampler,
        batch_size=8,
        shuffle=True,
        drop_last=True)
    """Extract node embeddings from the trained model"""
    model.eval()
    for it, (input_nodes, output_nodes, blocks) in enumerate(
        dataloader
    ):
        with torch.no_grad():
            node_embedding = model(blocks).detach().cpu().numpy()
    trip_od = torch.from_numpy(odflows[condition]) # src,dst,cnt(,dis_m)
    indices_o = [torch.where(output_nodes == b)[0] for b in trip_od[:,0]]
    flattened_o = torch.cat(indices_o).cpu().numpy()
    indices_d = [torch.where(output_nodes == b)[0] for b in trip_od[:,1]]
    flattened_d = torch.cat(indices_d).cpu().numpy()
    # construct edge feature
    scaled_dism = odflows[condition][:,3] #/ node_embedding.max()
    X = np.concatenate([node_embedding[flattened_o], node_embedding[flattened_d],scaled_dism.reshape(-1,1)], axis=1) 
    y = odflows[condition][:,2]
    return X, y

def train_LGBM(train_args, logger):
    # Load the best model
    device = torch.device(train_args.device)
    model_state_file = './ckpt/{}_layers{}_emb{}.pth'.format(train_args.log.strip('log/').strip('.log'),train_args.num_hidden_layers, train_args.embedding_size)
    checkpoint = torch.load(model_state_file)
    
    # Load data
    region = train_args.region
    data = utils.load_nids_dataset(year=train_args.year, node_feats_path=train_args.node_feats_path, region=region)
    
    # Get node IDs
    train_nids = data['train_nids']
    valid_nids = data['valid_nids']
    test_nids = data['test_nids']
    odflows = data['odflows']
    
    # Build graph
    node_feats = data['node_feats']
    ct_adj = data['weighted_adjacency']
    g = utils.build_graph_from_matrix(ct_adj, node_feats.astype(np.float32), device)
    
    # Initialize and load model
    model = MyModelBlock(data['num_nodes'], 
                        in_dim=node_feats.shape[1], 
                        h_dim=train_args.embedding_size, 
                        num_hidden_layers=train_args.num_hidden_layers,
                        init_noise_sigma=np.std(np.log10(odflows[np.isin(odflows[:, 0], train_nids)][:, 2])),
                        device=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
    # Extract embeddings for all nodes
    logger.info("Extracting embeddings...")
    X_train, y_train = extract_embeddings(model, g, train_nids, odflows, train_args, device)
    X_test, y_test = extract_embeddings(model, g, test_nids, odflows, train_args, device)

    # Train Random Forest
    logger.info("Training LGBM model...")
    lgbm_params = {'max_depth':10}
    gbm = lgb.LGBMRegressor( **lgbm_params,
                seed=42 #
                )
    gbm.fit(X_train, y_train)
    y_gbm = gbm.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_gbm))
    mae = mean_absolute_error(y_test, y_gbm)
    
    logger.info(f"LGBM Test |  RMSE: {rmse:.4f} - MAE: {mae:.4f} - CPC: {utils.CPC_(y_test, y_gbm)  :.4f}")


if __name__ == "__main__":
    args = parser.parse_args()
        # logger
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(args.log, mode='a'), logging.StreamHandler()])
    logger = logging.getLogger('#layers {}, emb {}'.format(args.num_hidden_layers, args.embedding_size))
    logger.setLevel(logging.DEBUG)
    # Train the GNN model
    train(args, logger)
    
    # Train and evaluate LGBM model
    train_LGBM(args, logger)  
