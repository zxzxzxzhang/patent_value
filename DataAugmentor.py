'''
Pre-release Notice

This repository contains code associated with our ongoing research project titled "Research on patent portfolio valuation based on Multi-SAGE-TechNexus model". The code is being made available for **review purposes only** and is subject to the following restrictions:

1. Non-commercial use only: This code may only be used for academic or non-commercial purposes.
2. No redistribution or modification**: Redistribution or modification of this code is not permitted until the associated research paper has been officially published.
3. Temporary access: The code in this repository is subject to updates and may change without notice until the final release.

After the publication of the corresponding research paper, we plan to release the code under a more permissive open-source license (e.g., MIT License).

For any questions or specific permissions, please contact zhangx2293@gmail.com with the subject "Pre-release Code Inquiry".

Written by Xiang Zhang
'''

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

        # Random edge addition
        num_edges_to_add = int(perturbation_ratio * num_edges)
        new_edges = torch.randint(0, num_nodes, (2, num_edges_to_add), device=data.edge_index.device)
        augmented_edge_index = torch.cat([data.edge_index, new_edges], dim=1)

        # Random edge removal
        num_edges_to_remove = int(perturbation_ratio * num_edges)
        mask = torch.ones(augmented_edge_index.size(1), dtype=torch.bool, device=augmented_edge_index.device)
        remove_indices = torch.randperm(augmented_edge_index.size(1), device=augmented_edge_index.device)[
                         :num_edges_to_remove]
        mask[remove_indices] = False
        augmented_edge_index = augmented_edge_index[:, mask]

        data.edge_index = augmented_edge_index

        return data

    def drop_edge(self, data, drop_prob):
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        keep_prob = 1 - drop_prob

        mask = torch.rand(num_edges, device=edge_index.device) < keep_prob
        edge_index = edge_index[:, mask]

        data.edge_index = edge_index

        return data

    def drop_node(self, data, drop_prob):
        num_nodes = data.num_nodes
        keep_prob = 1 - drop_prob

        keep_mask = torch.rand(num_nodes, device=data.x.device) < keep_prob
        keep_indices = keep_mask.nonzero(as_tuple=False).view(-1)

        data.x = data.x[keep_indices]
        data.y = data.y[keep_indices]

        if hasattr(data, 'train_mask'):
            data.train_mask = data.train_mask[keep_indices]
        if hasattr(data, 'test_mask'):
            data.test_mask = data.test_mask[keep_indices]

        node_idx_mapping = torch.zeros(num_nodes, dtype=torch.long, device=data.x.device)
        node_idx_mapping[keep_indices] = torch.arange(len(keep_indices), device=data.x.device)

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
        Subgraph Sampling

        Parameters:
        - num_hops: Number of hops to specify the k-hop subgraph
        - num_nodes: Number of nodes to include in the sampled subgraph
        """

        total_nodes = data.num_nodes

        seed_nodes = torch.randperm(total_nodes)[:num_nodes]


        node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
            seed_nodes, num_hops, data.edge_index, relabel_nodes=True, num_nodes=total_nodes)

        data.x = data.x[node_idx]
        data.y = data.y[node_idx]
        data.edge_index = edge_index

        if hasattr(data, 'train_mask'):
            data.train_mask = data.train_mask[node_idx]
        if hasattr(data, 'test_mask'):
            data.test_mask = data.test_mask[node_idx]

        return data

    def virtual_adversarial_training(self, model, data, xi=1e-6, eps=3.0):
        """
        Virtual Adversarial Training (VAT), enhances the model by adding adversarial perturbations
        to the input node features.

        Parameters:
        - model: The current graph neural network model
        - data: The data
        - xi: Initial perturbation coefficient
        - eps: Perturbation magnitude
        """

        model.eval()
        x = data.x.clone().detach()
        x.requires_grad = True

        d = torch.randn_like(x)
        d = xi * F.normalize(d, p=2, dim=1)

        out = model(data)
        out = out[data.train_mask]
        y_pred = F.softmax(out, dim=1).detach()

        data_adv = data.clone()
        data_adv.x = x + d
        out_adv = model(data_adv)
        out_adv = out_adv[data.train_mask]
        y_adv = F.log_softmax(out_adv, dim=1)

        vat_loss = F.kl_div(y_adv, y_pred, reduction='batchmean')

        return vat_loss

    def apply_attribute_masking(self, data, mask_prob=0.05):

        x = data.x.clone()
        mask = torch.rand(x.size(), device=x.device) < mask_prob
        x[mask] = 0
        data.x = x
        return data

    def apply_graph_mixup(self, data, alpha=0.2):
        """
        Graph Mixup, randomly selects two nodes and mixes their features and labels.

        Parameters:
        - alpha: Parameter controlling the mixing ratio
        """

        num_nodes = data.num_nodes
        idx1 = torch.randint(0, num_nodes, (num_nodes,), device=data.x.device)
        idx2 = torch.randint(0, num_nodes, (num_nodes,), device=data.x.device)
        lam = np.random.beta(alpha, alpha)
        data.x = lam * data.x[idx1] + (1 - lam) * data.x[idx2]
        data.y = lam * data.y[idx1] + (1 - lam) * data.y[idx2]
        return data
