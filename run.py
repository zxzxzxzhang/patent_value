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
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import umap
from sklearn.preprocessing import StandardScaler
from DataLoader import generate_text_embeddings, extract_citation_edges, prepare_graph_data
import random
from DataAugmentor import DataAugmentor
from models.model_Gated import HierarchicalBidirectionalSAGE_Gated
from models.model_GCN import GCN
from models.model_GIN import GIN
from models.model_GAT import GAT

data_name = 'pbat_cn_low.xlsx'
features = ['公开(公告)号',"技术覆盖范围", "引证次数", "被引证次数", "文献页数", "发明人数量",
                "科学关联性", "他引率", "权利要求数量", "首权字数", "无效次数",
                "专利有效性", "诉讼次数", "审查时长", "剩余寿命", "专利年龄",
                "申请人类型",  "商业化次数", "同族国家数", "质押次数", '专利价值']
citation = True
cited = False
umap_dimension = 20
epochs = 3000
centrality_types= ['degree']
noise_scale = 0.01 # 节点特征扰动
drop_edge_ratio =  0.18  # 边丢弃
drop_node_ratio =  0.01  # 节点丢弃
graph_mixup = True  # Graph Mixup
subgraph_sampling = False  # 子图采样
attribute_masking = False  # 属性遮掩
vat = True


dropout = 0.3
hidden_channels1 = 1024  # 隐藏层的维度，可以根据需要调整
hidden_channels2 = 768
lr = 0.0015
graph_model = GIN

def set_seed(seed):
    torch.manual_seed(seed)  # 设置CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，设置所有GPU的随机种子
    np.random.seed(seed)  # 设置Numpy的随机种子
    random.seed(seed)  # 设置Python中的随机种子

    # 确保GPU计算中的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 使用一个固定的种子来确保结果可重复
set_seed(38)

df1 = pd.read_excel(data_name)
# 选取数值特征
num_data = df1[features]

columns = ['标题 (英文)', '摘要 (英文)', '权利要求 (英文)']
embeddings = generate_text_embeddings(df1, columns, merge_before_training=False)

print('embedding Done')
print('embeddings shape :',embeddings.shape)

# 创建 UMAP 对象
umap_model = umap.UMAP(n_components=umap_dimension, random_state=42)
umap_embeddings = umap_model.fit_transform(embeddings)
print('UMAP Done')
print('umap_embeddings shape:', umap_embeddings.shape)
num_data_clean = num_data.drop(columns=['公开(公告)号','专利价值'], axis=1).values

concatenated = np.concatenate((num_data_clean, umap_embeddings), axis=1)
#2.对数值特征进行标准化
scaler = StandardScaler()
X_concatenated = scaler.fit_transform(concatenated)
print('X_concatenated.shape :',X_concatenated.shape)

# 假设df1是原始数据，包含‘公开(公告)号’，‘[标]原始申请(专利权)人’ 和 ‘引用专利’
# 示例数据处理 - 构建基于发明人的合作网络
df1['专利序号'] = df1['公开(公告)号']
prices = df1['专利价值'].tolist()

# 提前收集所有专利序号，减少重复检查
all_patents_set = set(df1['专利序号'].values)  # 使用集合提升查找效率

citation_edges = extract_citation_edges(df1, '引用专利', all_patents_set)
cited_edges = extract_citation_edges(df1, '被引用专利', all_patents_set)

# 创建有向图
DG = nx.DiGraph()

# 批量添加所有专利节点
DG.add_nodes_from(all_patents_set)  # 一次性添加所有专利节点

if citation:
    DG.add_edges_from(citation_edges)  # 有向引用关系边
if cited:
    DG.add_edges_from(cited_edges)  # 有向引用关系边

# 打印有向图的节点数量
print(f"Number of nodes in directed graph DG: {DG.number_of_nodes()}")

# 调用函数并移动数据到 GPU
data_with_citations = prepare_graph_data(DG, X_concatenated, prices, centrality_types=centrality_types, train_ratio=0.8, seed=46)
device = torch.device("cuda")
data = data_with_citations.to(device)

data_augmentor = DataAugmentor(
    noise_scale=noise_scale,            # 节点特征扰动
    drop_edge_ratio=drop_edge_ratio,        # 边丢弃
    drop_node_ratio=drop_node_ratio,        # 节点丢弃
    #mask_feature_ratio=0.10,     # 特征遮掩
    graph_mixup=graph_mixup,            # Graph Mixup
    subgraph_sampling=subgraph_sampling,      # 子图采样
    attribute_masking=attribute_masking,      # 属性遮掩
    vat=vat                    # 虚拟对抗训练（VAT）
)

# 定义RMSE、MAE、MAPE和R²的计算函数
def calculate_metrics(preds, targets):
    mse = F.mse_loss(preds, targets).item()
    rmse = torch.sqrt(F.mse_loss(preds, targets)).item()  # RMSE
    mae = F.l1_loss(preds, targets).item()  # MAE
    mape = (torch.mean(torch.abs((targets - preds) / targets)) * 100).item()  # MAPE
    ss_res = torch.sum((targets - preds) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1 - ss_res / ss_tot  # R²

    return rmse, mae, mape, r2

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()

    if data_augmentor.vat:
        augmented_data, vat_loss = data_augmentor.augment(data, model=model)
    else:
        augmented_data = data_augmentor.augment(data)

    augmented_data = augmented_data.to(device)  # 将增强数据移到GPU

    out = model(augmented_data)  # 使用增强后的数据进行训练
    out = out.squeeze()

        # 只对训练集上的节点计算损失
    loss = F.mse_loss(out[augmented_data.train_mask], augmented_data.y[augmented_data.train_mask])
    loss.backward()
    optimizer.step()

    # 计算 RMSE, MAE, MAPE 和 R²
    rmse, mae, mape, r2 = calculate_metrics(out[augmented_data.train_mask], augmented_data.y[augmented_data.train_mask])

    return loss.item(), rmse, mae, mape, r2

# 测试函数
def test():
    model.eval()
    with torch.no_grad():
        out = model(data)
        out = out.squeeze()
        # 只对测试集上的节点计算损失
        loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])

        # 计算 RMSE, MAE, MAPE 和 R²
        rmse, mae, mape, r2 = calculate_metrics(out[data.test_mask], data.y[data.test_mask])

    return loss.item(), rmse, mae, mape, r2

# 初始化模型和优化器
in_channels = data.num_node_features  # 输入维度（节点特征的维度）
out_channels = 1  # 输出维度为1（预测专利价值）
best_epoch = -1

model = graph_model(in_channels, hidden_channels1, hidden_channels2,out_channels, dropout=dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
# 记录每一轮的训练损失和测试损失
train_losses = []
test_losses = []
train_metrics = []
test_metrics = []

# 初始化最低测试损失和各评估指标为一个很大的值
min_test_loss = float('inf')
best_test_metrics = {'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'mape': float('inf'), 'r2': -float('inf')}


for epoch in range(epochs):
    train_loss, train_rmse, train_mae, train_mape, train_r2 = train()
    test_loss, test_rmse, test_mae, test_mape, test_r2 = test()

    # 记录损失和评估指标
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_metrics.append((train_rmse, train_mae, train_mape, train_r2))
    test_metrics.append((test_rmse, test_mae, test_mape, test_r2))

    # 更新最低测试损失和最佳评估指标
    if test_loss < min_test_loss:
        min_test_loss = test_loss
        best_epoch = epoch
        best_test_metrics['rmse'] = test_rmse
        best_test_metrics['mae'] = test_mae
        best_test_metrics['mape'] = test_mape
        best_test_metrics['r2'] = test_r2

    # 每100个epoch打印一次训练和测试的结果
    if epoch % 100 == 0:
        print(f'Epoch: {epoch:04d}')
        print(f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
        print(f'Train Metrics - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.2f}%, R²: {train_r2:.4f}')
        print(f'Test  Metrics - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}')
        print('-' * 80)

# 打印测试集上的最佳结果
print("Best Test Metrics:")
print(f'MSE: {min_test_loss:.4f}(achieved at epoch {best_epoch})')
print(f'RMSE: {best_test_metrics["rmse"]:.4f}')
print(f'MAE: {best_test_metrics["mae"]:.4f}')
print(f'MAPE: {best_test_metrics["mape"]:.2f}%')
print(f'R²: {best_test_metrics["r2"]:.4f}')

# 绘制训练和测试损失变化图
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_losses, label="Train Loss (MSE)", color="blue")
plt.plot(range(epochs), test_losses, label="Test Loss (MSE)", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Train and Test Loss Over Epochs")
plt.legend()
plt.show()
plt.savefig('train_test_loss_over_epochs.png', dpi=1000)