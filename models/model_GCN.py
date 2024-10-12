# --------------------------------------------------------
# Sage_Gated
# Copyright (c) 2024 Xiang Zhang
# All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Xiang Zhang
# --------------------------------------------------------

import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        # 定义卷积层
        self.conv1 = GCNConv(in_channels, hidden_channels1)  # 正向卷积
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)  # 反向卷积
        self.conv3 = GCNConv(hidden_channels2, out_channels)  # 正向卷积

        # dropout率
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 第1层：正向传播
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第2层：反向传播
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第3层：正向传播
        x = self.conv3(x, edge_index)

        return x
