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
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class MultiResolutionGatedLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, hops=[1, 2, 3, 4]):
        super(MultiResolutionGatedLayer, self).__init__()
        self.num_heads = num_heads
        self.hops = hops  # hops
        self.convs = torch.nn.ModuleList([SAGEConv(in_channels, out_channels) for _ in range(num_heads)])
        self.gate = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        head_outs = []
        for i, conv in enumerate(self.convs):
            if self.hops[i] == 1:
                out = conv(x, edge_index)
            else:
                current_edge_index = self.multi_hop_edges(edge_index, x.size(0), hop=self.hops[i])
                out = conv(x, current_edge_index)
            head_outs.append(out)

        out = torch.mean(torch.stack(head_outs, dim=0), dim=0)
        # Gated
        gate = torch.sigmoid(self.gate(out))
        return out * gate

    def multi_hop_edges(self, edge_index, num_nodes, hop=2):
        """Expand to multi-hop neighbors"""
        device = edge_index.device
        adj_matrix = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1), device=device), (num_nodes, num_nodes)).to(device)
        adj_matrix_power = adj_matrix
        for _ in range(hop - 1):
            adj_matrix_power = torch.sparse.mm(adj_matrix, adj_matrix_power)
        new_edge_index = adj_matrix_power.coalesce().indices().to(device)
        return new_edge_index


class Multi_SAGE_Gated(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels, num_heads=4, dropout=0.5):
        super(Multi_SAGE_Gated, self).__init__()

        self.num_heads = num_heads
        self.gated_conv1 = MultiResolutionGatedLayer(in_channels, hidden_channels1, num_heads=self.num_heads, hops=[1, 2, 3, 4])
        self.gated_conv2 = MultiResolutionGatedLayer(hidden_channels1, hidden_channels2, num_heads=self.num_heads, hops=[1, 2, 3, 4])
        self.gated_conv3 = MultiResolutionGatedLayer(hidden_channels2, out_channels, num_heads=self.num_heads, hops=[1, 2, 3, 4])

        # dropout
        self.dropout = dropout

        # LayerNorm
        self.layer_norm1 = torch.nn.LayerNorm(hidden_channels1)
        self.layer_norm2 = torch.nn.LayerNorm(hidden_channels2)

        # Residual
        self.fc_residual = torch.nn.Linear(in_channels, hidden_channels1)
        self.fc_residual_out = torch.nn.Linear(in_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Residual
        residual_input = x

        # layer1: Forward + Residual
        x = self.gated_conv1(x, edge_index)
        x = self.layer_norm1(x)
        x = F.gelu(x)

        residual_x = self.fc_residual(residual_input)
        x = x + residual_x
        x = F.dropout(x, p=self.dropout, training=self.training)

        # layer2: Backward
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        x = self.gated_conv2(x, reversed_edge_index)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # layer3: Forward + Residual
        x = self.gated_conv3(x, edge_index)

        residual_x_out = self.fc_residual_out(residual_input)
        x = x + residual_x_out

        return x
