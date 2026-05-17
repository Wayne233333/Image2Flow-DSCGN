import pandas as pd
import numpy as np
import torch
import logging 

from dataset import utils
from modules.gnn import MyModelBlock
import dgl
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='log/OD_LA.log') # model state of city A 
parser.add_argument('--node_feats_path', type=str, default='../data/Vis/train_on_LA_2020.csv') # visual features of city B
parser.add_argument('--region', type=str, default='LA') # city B
parser.add_argument('--year', type=str, default='2020') 
parser.add_argument('--device', type=str, default = 'cuda:0')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default = 1e-5)
parser.add_argument('--grad_norm', type=float, default=1.0)
parser.add_argument('--evaluate_every', type=int, default=5)

parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--embedding_size', type=int, default=512)

def test(test_args):
    # device
    device = torch.device(test_args.device)

    # Setup logging
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(test_args.log, mode='a'), logging.StreamHandler()])
    logger = logging.getLogger('#layers{}_emb{}'.format(test_args.num_hidden_layers, test_args.embedding_size))
    logger.setLevel(logging.DEBUG)

    # load data
    region = test_args.region
    data = utils.load_nids_dataset(year=test_args.year,node_feats_path=test_args.node_feats_path, region=region)

    logger.info("-----------------------------------------")
    logger.info(f"Starting testing with parameters:")
    logger.info(f"Region: {region}")
    logger.info(f"Year: {test_args.year}")
    logger.info(f"Node features path: {test_args.node_feats_path}")
    logger.info(f"Number of hidden layers: {test_args.num_hidden_layers}")
    logger.info(f"Embedding size: {test_args.embedding_size}")
    logger.info("-----------------------------------------")

    logger.info("----------------------------------------- "+region+" "+test_args.year+" Test all_nids")
    all_nids = data['all_nids'] 

    odflows = data['odflows']
    all_nids_d = np.unique(odflows[np.isin(odflows[:, 0], all_nids)][:,1]).astype('int64')
    all_nids_od =  np.unique(np.append(all_nids, all_nids_d, axis=0))
    
    node_feats = data['node_feats']
    potential = data['potential']

    ct_adj = data['weighted_adjacency']
    num_nodes = data['num_nodes']

    model = MyModelBlock(num_nodes, in_dim = node_feats.shape[1], h_dim = test_args.embedding_size, num_hidden_layers=test_args.num_hidden_layers, device=device)
    g = utils.build_graph_from_matrix(ct_adj, node_feats.astype(np.float32), potential, device)  

    model.to(device)
       
    g.to(device)

    # minibatch
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(test_args.num_hidden_layers+1)
    # sampler = dgl.dataloading.MultiLayerNeighborSampler([200] * (test_args.num_hidden_layers + 1))

    test_dataloader = dgl.dataloading.DataLoader(
        g, torch.from_numpy(all_nids_od).to(device), sampler,
        batch_size=len(all_nids_od),
        shuffle=True,
        drop_last=False,
        num_workers=0) # 

    
    # training recorder
    model_state_file = './ckpt/{}_layers{}_emb{}.pth'.format(test_args.log.strip('log/').strip('.log'),test_args.num_hidden_layers, test_args.embedding_size)
    
    log_base_name = os.path.basename(test_args.log).replace(".log", "")
    prefix = f"{log_base_name}_layers{test_args.num_hidden_layers}_emb{test_args.embedding_size}"
    train_on = log_base_name.replace("bands3", "").replace("bands6", "")
      
    # Test:
    logger.info(f"Loading model from {model_state_file}")
    model.load_state_dict(torch.load(model_state_file)['state_dict']) 
    model.eval()
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            test_dataloader
    ):
        with torch.no_grad():
            node_embedding = model(blocks).detach().cpu().numpy()
            origin_nids = output_nodes.cpu()[np.isin(output_nodes.cpu(), all_nids)]

            # Filter OD flows: only keep flows whose origin is in all_nids AND destination is in output_nodes
            condition1 = np.isin(odflows[:, 0], origin_nids)
            condition2 = np.isin(odflows[:, 1], output_nodes.cpu())
            combined_condition = condition1 & condition2

            trip_od_test = torch.from_numpy(odflows[combined_condition][:, :2].astype(np.int64)).to(device)

            trip_volume_test = torch.from_numpy(odflows[combined_condition][:, 2].astype(float)).to(device)
            log_trip_volume_test = utils.log_transform(trip_volume_test)

        rmse, mae, cpc = utils.evaluateOutput(model, blocks, trip_od_test, trip_volume_test, output_nodes, region, prefix, train_on)
        
        logger.info("-----------------------------------------")
        logger.info(f'Test | Bilinear Decoder '
                 f'RMSE: {rmse:.4f} - MAE: {mae:.4f} - '
                 f'CPC: {cpc:.4f}')

    # LGBM section: use the same filtered OD flows for consistency
    indices_o = [torch.where(output_nodes == b)[0] for b in trip_od_test[:,0]]
    flattened_o = torch.cat(indices_o).cpu().numpy()
    indices_d = [torch.where(output_nodes == b)[0] for b in trip_od_test[:,1]]
    flattened_d = torch.cat(indices_d).cpu().numpy()
    scaled_dism_test = odflows[combined_condition][:, 3]
    # construct edge feature

    X_test = np.concatenate([node_embedding[flattened_o], node_embedding[flattened_d],scaled_dism_test.reshape(-1,1)], axis=1)
    y_test = odflows[combined_condition][:, 2]

    with open('./models/lgbm_{}.txt'.format(args.log.strip('log/').strip('.log')), 'rb') as file:
        gbm = pickle.load(file)
    y_gbm = gbm.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_gbm))
    mae = mean_absolute_error(y_test, y_gbm)
    logger.info(f"Test - | LGBM Decoder RMSE: {rmse:.4f} - MAE: {mae:.4f} - CPC: {utils.CPC_(y_test, y_gbm)  :.4f}")


if __name__ == "__main__":
    args = parser.parse_args()
    
    test(args)