# --------------------------------------------------------
# Sage_Gated
# Copyright (c) 2021 Xiang Zhang
# All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Xiang Zhang
# --------------------------------------------------------

import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
import numpy as np

# DataAugmentor
class DataAugmentor:
    def __init__(self, noise_scale=0.0, edge_perturbation_ratio=0.0, drop_edge_ratio=0.0,
                 drop_node_ratio=0.0, mask_feature_ratio=0.0, subgraph_sampling=False,
                 graph_mixup=False, attribute_masking=False, vat=False):
        self.noise_scale = noise_scale
        self.edge_perturbation_ratio = edge_perturbation_ratio
        self.drop_edge_ratio = drop_edge_ratio
        self.drop_node_ratio = drop_node_ratio
        self.mask_feature_ratio = mask_feature_ratio
        self.subgraph_sampling = subgraph_sampling
        self.graph_mixup = graph_mixup
        self.attribute_masking = attribute_masking
        self.vat = vat

    def augment(self, data, model=None):
        augmented_data = data.clone()

        if self.noise_scale > 0:
            noise = torch.randn_like(augmented_data.x) * self.noise_scale
            augmented_data.x = augmented_data.x + noise

        if self.edge_perturbation_ratio > 0:
            augmented_data = self.edge_perturbation(augmented_data, self.edge_perturbation_ratio)

        if self.drop_edge_ratio > 0:
            augmented_data = self.drop_edge(augmented_data, self.drop_edge_ratio)

        if self.drop_node_ratio > 0:
            augmented_data = self.drop_node(augmented_data, self.drop_node_ratio)

        if self.mask_feature_ratio > 0:
            augmented_data = self.mask_feature(augmented_data, self.mask_feature_ratio)

        if self.subgraph_sampling:
            augmented_data = self.apply_subgraph_sampling(augmented_data)

        if self.graph_mixup:
            augmented_data = self.apply_graph_mixup(augmented_data)

        if self.attribute_masking:
            augmented_data = self.apply_attribute_masking(augmented_data)

        if self.vat and model is not None:
            vat_loss = self.virtual_adversarial_training(model, augmented_data)
            return augmented_data, vat_loss

        return augmented_data

    def edge_perturbation(self, data, perturbation_ratio):
        num_edges = data.edge_index.size(1)
        num_nodes = data.num_nodes

        # 随机添加边
        num_edges_to_add = int(perturbation_ratio * num_edges)
        new_edges = torch.randint(0, num_nodes, (2, num_edges_to_add), device=data.edge_index.device)
        augmented_edge_index = torch.cat([data.edge_index, new_edges], dim=1)

        # 随机删除边
        num_edges_to_remove = int(perturbation_ratio * num_edges)
        mask = torch.ones(augmented_edge_index.size(1), dtype=torch.bool, device=augmented_edge_index.device)
        remove_indices = torch.randperm(augmented_edge_index.size(1), device=augmented_edge_index.device)[
                         :num_edges_to_remove]
        mask[remove_indices] = False
        augmented_edge_index = augmented_edge_index[:, mask]

        # 更新边索引
        data.edge_index = augmented_edge_index

        return data

    def drop_edge(self, data, drop_prob):
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        keep_prob = 1 - drop_prob

        # 随机选择要保留的边
        mask = torch.rand(num_edges, device=edge_index.device) < keep_prob
        edge_index = edge_index[:, mask]

        # 更新边索引
        data.edge_index = edge_index

        return data

    def drop_node(self, data, drop_prob):
        num_nodes = data.num_nodes
        keep_prob = 1 - drop_prob

        # 随机选择要保留的节点
        keep_mask = torch.rand(num_nodes, device=data.x.device) < keep_prob
        keep_indices = keep_mask.nonzero(as_tuple=False).view(-1)

        # 更新节点特征和标签
        data.x = data.x[keep_indices]
        data.y = data.y[keep_indices]

        # 更新训练和测试掩码
        if hasattr(data, 'train_mask'):
            data.train_mask = data.train_mask[keep_indices]
        if hasattr(data, 'test_mask'):
            data.test_mask = data.test_mask[keep_indices]

        # 创建新的节点索引映射
        node_idx_mapping = torch.zeros(num_nodes, dtype=torch.long, device=data.x.device)
        node_idx_mapping[keep_indices] = torch.arange(len(keep_indices), device=data.x.device)

        # 更新边索引
        edge_index = data.edge_index
        src, dst = edge_index
        src_mask = keep_mask[src]
        dst_mask = keep_mask[dst]
        edge_mask = src_mask & dst_mask
        new_src = node_idx_mapping[src[edge_mask]]
        new_dst = node_idx_mapping[dst[edge_mask]]
        data.edge_index = torch.stack([new_src, new_dst], dim=0)

        return data

    def mask_feature(self, data, mask_prob):
        x = data.x
        mask = torch.rand_like(x) > mask_prob
        data.x = x * mask.float()
        return data

    def apply_subgraph_sampling(self, data, num_hops=2, num_nodes=1000):
        """
        子图采样，随机采样指定数量的节点，生成新的子图。

        参数:
        - num_hops: 指定采样子图的k跳数
        - num_nodes: 采样的子图中包含的节点数量
        """
        total_nodes = data.num_nodes
        # 随机选择种子节点
        seed_nodes = torch.randperm(total_nodes)[:num_nodes]

        # 获取子图
        node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
            seed_nodes, num_hops, data.edge_index, relabel_nodes=True, num_nodes=total_nodes)

        # 更新数据
        data.x = data.x[node_idx]
        data.y = data.y[node_idx]
        data.edge_index = edge_index

        # 更新掩码
        if hasattr(data, 'train_mask'):
            data.train_mask = data.train_mask[node_idx]
        if hasattr(data, 'test_mask'):
            data.test_mask = data.test_mask[node_idx]

        return data

    def virtual_adversarial_training(self, model, data, xi=1e-6, eps=3.0):
        """
        虚拟对抗训练（VAT），通过对输入节点特征添加对抗性扰动来增强模型。

        参数:
        - model: 当前的图神经网络模型
        - data: 数据
        - xi: 初始扰动系数
        - eps: 扰动幅度
        """
        model.eval()
        x = data.x.clone().detach()
        x.requires_grad = True

        # 初始扰动
        d = torch.randn_like(x)
        d = xi * F.normalize(d, p=2, dim=1)

        # 前向传播
        out = model(data)
        out = out[data.train_mask]
        y_pred = F.softmax(out, dim=1).detach()

        # 添加扰动并前向传播
        data_adv = data.clone()
        data_adv.x = x + d
        out_adv = model(data_adv)
        out_adv = out_adv[data.train_mask]
        y_adv = F.log_softmax(out_adv, dim=1)

        # 计算虚拟对抗损失
        vat_loss = F.kl_div(y_adv, y_pred, reduction='batchmean')

        return vat_loss

    def apply_attribute_masking(self, data, mask_prob=0.05):
        """
        随机遮掩节点特征中的部分元素。

        参数:
        - mask_prob: 遮掩的比例
        """
        x = data.x.clone()
        mask = torch.rand(x.size(), device=x.device) < mask_prob
        x[mask] = 0
        data.x = x
        return data

    def apply_graph_mixup(self, data, alpha=0.2):
        """
        Graph Mixup，随机选择两个节点，将它们的特征和标签进行混合。

        参数:
        - alpha: 控制混合比例的参数
        """
        num_nodes = data.num_nodes
        idx1 = torch.randint(0, num_nodes, (num_nodes,), device=data.x.device)
        idx2 = torch.randint(0, num_nodes, (num_nodes,), device=data.x.device)
        lam = np.random.beta(alpha, alpha)
        data.x = lam * data.x[idx1] + (1 - lam) * data.x[idx2]
        data.y = lam * data.y[idx1] + (1 - lam) * data.y[idx2]
        return data