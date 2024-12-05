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
import networkx as nx
import torch

def calculate_centrality(DG, centrality_types):
    centrality_tensors = []

    if 'degree' in centrality_types:
        # degree
        degree_centrality = {node: val for node, val in DG.degree()}
        degree_tensor = torch.tensor([degree_centrality[node] for node in DG.nodes()], dtype=torch.float).view(-1, 1)
        centrality_tensors.append(degree_tensor)

    if 'betweenness' in centrality_types:
        # betweenness
        betweenness_centrality = nx.betweenness_centrality(DG)
        betweenness_tensor = torch.tensor([betweenness_centrality[node] for node in DG.nodes()],
                                          dtype=torch.float).view(-1, 1)
        centrality_tensors.append(betweenness_tensor)

    if 'closeness' in centrality_types:
        # closeness
        closeness_centrality = nx.closeness_centrality(DG)
        closeness_tensor = torch.tensor([closeness_centrality[node] for node in DG.nodes()], dtype=torch.float).view(-1,
                                                                                                                     1)
        centrality_tensors.append(closeness_tensor)

    if not centrality_tensors:
        raise ValueError(f"At least one valid type of centrality is required: 'degree', 'betweenness', 或 'closeness'")

    centrality_tensor_combined = torch.cat(centrality_tensors, dim=1)

    return centrality_tensor_combined


def augment_with_centrality(x, DG, centrality_types):
    centrality_tensor = calculate_centrality(DG, centrality_types)
    x_augmented = torch.cat([x, centrality_tensor], dim=1)

    return x_augmented
