# --------------------------------------------------------
# Sage_Gated
# Copyright (c) 2024 Xiang Zhang
# All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Xiang Zhang
# --------------------------------------------------------

import torch
from torch_geometric.nn import GINConv
import torch.nn.functional as F

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels, num_layers=3, dropout=0.5):
        super(GIN, self).__init__()
        self.num_layers = num_layers

        # 定义GIN卷积层
        self.conv1 = GINConv(torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels1), torch.nn.ReLU()))
        self.conv2 = GINConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels1, hidden_channels2), torch.nn.ReLU()))
        self.conv3 = GINConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels2, out_channels), torch.nn.ReLU()))

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
