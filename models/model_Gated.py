# --------------------------------------------------------
# Sage_Gated
# Copyright (c) 2024 Xiang Zhang
# All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Xiang Zhang
# --------------------------------------------------------

import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F


class GatedLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedLayer, self).__init__()
        self.conv = SAGEConv(in_channels, out_channels)
        self.gate = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        gate = torch.sigmoid(self.gate(out))
        return out * gate

class HierarchicalBidirectionalSAGE_Gated(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels, dropout=0.5):
        super(HierarchicalBidirectionalSAGE_Gated, self).__init__()

        # 定义门控卷积层
        self.gated_conv1 = GatedLayer(in_channels, hidden_channels1)
        self.gated_conv2 = GatedLayer(hidden_channels1, hidden_channels2)
        self.gated_conv3 = GatedLayer(hidden_channels2, out_channels)

        # dropout率
        self.dropout = dropout

        # 添加LayerNorm层
        self.layer_norm1 = torch.nn.LayerNorm(hidden_channels1)
        self.layer_norm2 = torch.nn.LayerNorm(hidden_channels2)

        # 残差连接
        self.fc_residual = torch.nn.Linear(in_channels, hidden_channels1)
        self.fc_residual_out = torch.nn.Linear(in_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 保存初始输入用于残差连接
        residual_input = x

        # 第1层：正向传播 + 残差连接
        x = self.gated_conv1(x, edge_index)
        x = self.layer_norm1(x)
        x = F.gelu(x)

        # 残差连接
        residual_x = self.fc_residual(residual_input)
        x = x + residual_x
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第2层：反向传播，不加残差连接
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        x = self.gated_conv2(x, reversed_edge_index)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第3层：正向传播 + 残差连接
        x = self.gated_conv3(x, edge_index)

        # 残差连接
        residual_x_out = self.fc_residual_out(residual_input)
        x = x + residual_x_out

        return x
