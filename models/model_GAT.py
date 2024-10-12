# --------------------------------------------------------
# Sage_Gated
# Copyright (c) 2021 Xiang Zhang
# All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Xiang Zhang
# --------------------------------------------------------

import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels, heads=1, num_layers=3, dropout=0.5):
        super(GAT, self).__init__()
        self.num_layers = num_layers

        # 定义GAT卷积层
        self.conv1 = GATConv(in_channels, hidden_channels1, heads=heads)  # 正向卷积
        self.conv2 = GATConv(hidden_channels1 * heads, hidden_channels2, heads=heads)  # 反向卷积
        self.conv3 = GATConv(hidden_channels2 * heads, out_channels, heads=heads)  # 正向卷积

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
