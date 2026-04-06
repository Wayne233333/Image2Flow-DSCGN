import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from torch.nn.modules.loss import _Loss

class MyModelBlock(nn.Module):
    # 【新增】pot_dim 参数，用于接收 POI 特征维度
    def __init__(self, num_nodes, in_dim, pot_dim, h_dim, weighted=False, init_noise_sigma=0.2, num_hidden_layers=1, device='cpu'):
        super().__init__()
        # 将 pot_dim 传递给 GAT
        self.gat = GAT(num_nodes, in_dim, pot_dim, h_dim, h_dim, num_hidden_layers, device) 
        self.bilinear = nn.Bilinear(h_dim, h_dim, 1)
        self.activation = nn.ReLU()
        self.criterion = BMCLoss(init_noise_sigma, weighted=weighted)

    def forward(self, g, blocks):
        # 这里的 g 是 DGL 的 mini-batch blocks
        return self.gat.forward(blocks)
    
    def get_loss(self, output_nodes, trip_od, scaled_trip_volume, blocks):
        node_embedding = self.forward(None, blocks)
        edge_prediction = self.predict_edge(node_embedding, trip_od, output_nodes)
        edge_predict_loss = self.criterion(edge_prediction.to(torch.float32), scaled_trip_volume.to(torch.float32))
        return edge_predict_loss 

    def predict_edge(self, node_embedding, trip_od, output_nodes):
        # 这里的逻辑保持不变
        rel_nids = torch.arange(len(output_nodes)).to(node_embedding.device)
        mapping = {nid.item(): i for i, nid in enumerate(output_nodes)}
        
        src_indices = torch.tensor([mapping[idx.item()] for idx in trip_od[:, 0]], device=node_embedding.device)
        dst_indices = torch.tensor([mapping[idx.item()] for idx in trip_od[:, 1]], device=node_embedding.device)
        
        src_emb = node_embedding[src_indices]
        dst_emb = node_embedding[dst_indices]
        
        res = self.bilinear(src_emb, dst_emb)
        return res

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma, weighted=False):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred, target, noise_var, weighted=False):
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var).detach()
    # gamma = 2
    if weighted:
        weights = (1 / target) # ** gamma
        loss = (loss * weights).mean()

    return loss

class GAT(nn.Module):
    def __init__(self, num_nodes, in_dim, pot_dim, h_dim, out_dim, num_hidden_layers, device):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.device = device
        
        # 【新增】势能转换层：将 POI 预测数据转换为势阱能量
        self.pot_encoder = nn.Sequential(
            nn.Linear(pot_dim, h_dim),
            nn.Tanh() # 限制势能范围
        )

        # 输入层
        self.layers.append(GATLayer(in_dim, h_dim, h_dim, h_dim))
        # 隐藏层
        for _ in range(num_hidden_layers - 1):
            self.layers.append(GATLayer(h_dim, h_dim, h_dim, h_dim))
        # 输出层
        self.layers.append(GATLayer(h_dim, out_dim, h_dim, h_dim))

    def forward(self, blocks):
        # 这里的 blocks 是采样后的子图
        h = blocks[0].srcdata['attr'].to(self.device)
        
        for i, layer in enumerate(self.layers):
            # 提取当前 block 的势能特征
            pot = blocks[i].dstdata['pot'].to(self.device)
            v_i = self.pot_encoder(pot) 
            
            # 将势能 v_i 传入卷积层
            h = layer(h, blocks[i], v_i)
        return h
    
class GATLayer(nn.Module):
    def __init__(self, in_ndim, out_ndim, in_edim, out_edim):
        super(GATLayer, self).__init__()
        self.fcV = nn.Linear(in_edim, out_edim, bias=False)
        self.fcW = nn.Linear(in_ndim, out_ndim, bias=False)
        self.fcU = nn.Linear(in_ndim, out_ndim, bias=False)
        self.attn_fc = nn.Linear(2 * out_ndim + out_edim, 1, bias=False)
        
        # 【DSGNN 关键项】势阱系数，控制 POI 对节点的吸引/排斥程度
        self.potential_gate = nn.Parameter(torch.ones(out_ndim))

    def edge_feat_func(self, edges):
        return {'t': self.fcV(edges.data['d'])}

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['t']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 邻居信息的加权聚合 (对应波函数的扩散项)
        z_neighbor = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        
        # 提取势阱信息 (对应薛定谔方程中的 V*psi 项)
        v_i = nodes.data['v_i']
        z_i = nodes.data['z_i']
        
        # DSGNN 融合逻辑：新状态 = 扩散项 + 势阱交互项
        # 我们使用哈达玛积 (Element-wise product) 让势能调制节点特征
        h_out = z_neighbor + (self.potential_gate * v_i * z_i)
        
        return {'h': h_out}

    def forward(self, h, g, v_i):
        with g.local_scope():
            # 基础特征变换
            z = self.fcW(h)
            g.srcdata['z'] = z
            g.dstdata['z_i'] = self.fcU(h[:g.num_dst_nodes()])
            # 【新增】将计算好的势能存入 dstdata
            g.dstdata['v_i'] = v_i
            
            # 计算注意力
            g.apply_edges(self.edge_attention)
            # 聚合
            g.update_all(self.message_func, self.reduce_func)
            return g.dstdata['h']