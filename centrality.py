# --------------------------------------------------------
# Sage_Gated
# Copyright (c) 2021 Xiang Zhang
# All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Xiang Zhang
# --------------------------------------------------------

import networkx as nx
import torch

def calculate_centrality(DG, centrality_types):
    centrality_tensors = []

    if 'degree' in centrality_types:
        # 计算度中心性
        degree_centrality = {node: val for node, val in DG.degree()}
        degree_tensor = torch.tensor([degree_centrality[node] for node in DG.nodes()], dtype=torch.float).view(-1, 1)
        centrality_tensors.append(degree_tensor)

    if 'betweenness' in centrality_types:
        # 计算介数中心性
        betweenness_centrality = nx.betweenness_centrality(DG)
        betweenness_tensor = torch.tensor([betweenness_centrality[node] for node in DG.nodes()],
                                          dtype=torch.float).view(-1, 1)
        centrality_tensors.append(betweenness_tensor)

    if 'closeness' in centrality_types:
        # 计算接近中心性
        closeness_centrality = nx.closeness_centrality(DG)
        closeness_tensor = torch.tensor([closeness_centrality[node] for node in DG.nodes()], dtype=torch.float).view(-1,
                                                                                                                     1)
        centrality_tensors.append(closeness_tensor)

    if not centrality_tensors:
        raise ValueError(f"至少需要一个有效的中心性类型: 'degree', 'betweenness', 或 'closeness'")

    centrality_tensor_combined = torch.cat(centrality_tensors, dim=1)

    return centrality_tensor_combined


def augment_with_centrality(x, DG, centrality_types):
    centrality_tensor = calculate_centrality(DG, centrality_types)
    x_augmented = torch.cat([x, centrality_tensor], dim=1)

    return x_augmented
