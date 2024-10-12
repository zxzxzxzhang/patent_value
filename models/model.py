# --------------------------------------------------------
# Sage_Gated
# Copyright (c) 2021 Xiang Zhang
# All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Xiang Zhang
# --------------------------------------------------------

import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class HierarchicalBidirectionalSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels, num_layers=3, dropout=0.5):
        super(HierarchicalBidirectionalSAGE, self).__init__()
        self.num_layers = num_layers

        # 定义卷积层
        self.conv1 = SAGEConv(in_channels, hidden_channels1)  # 正向卷积
        self.conv2 = SAGEConv(hidden_channels1, hidden_channels2)  # 反向卷积
        self.conv3 = SAGEConv(hidden_channels2, out_channels)  # 正向卷积

        # dropout率
        self.dropout = dropout

        # 添加LayerNorm层
        self.layer_norm1 = torch.nn.LayerNorm(hidden_channels1)
        self.layer_norm2 = torch.nn.LayerNorm(hidden_channels2)

        # 残差连接
        self.fc_residual = torch.nn.Linear(in_channels, hidden_channels1)  # 用于第1层残差连接
        self.fc_residual_out = torch.nn.Linear(in_channels, out_channels)  # 用于第3层残差连接

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 保存初始输入用于残差连接
        residual_input = x

        # 第1层：正向传播 + 残差连接
        x = self.conv1(x, edge_index)
        x = self.layer_norm1(x)
        x = F.gelu(x)

        # 残差连接：将输入的维度通过全连接层转换成 hidden_channels
        residual_x = self.fc_residual(residual_input)
        x = x + residual_x
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第2层：反向传播，不加残差连接
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)  # 反转边
        x = self.conv2(x, reversed_edge_index)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第3层：正向传播 + 残差连接
        x = self.conv3(x, edge_index)

        # 残差连接：将初始输入维度转换成输出维度
        residual_x_out = self.fc_residual_out(residual_input)
        x = x + residual_x_out

        return x