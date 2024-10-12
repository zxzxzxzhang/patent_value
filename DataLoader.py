# --------------------------------------------------------
# Sage_Gated
# Copyright (c) 2021 Xiang Zhang
# All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Xiang Zhang
# --------------------------------------------------------

from sentence_transformers import SentenceTransformer
import numpy as np
from centrality import augment_with_centrality
import torch
from torch_geometric.data import Data

# embedding
def generate_text_embeddings(df, columns, model_name='distilbert_finetuned', merge_before_training=True):
    """
    根据指定的列生成文本的向量表示，支持合并列后训练或逐列训练后再合并。

    参数:
    - df: 输入的DataFrame
    - columns: 需要处理的列名列表
    - model_name: 使用的预训练模型名称（默认为'distilbert_finetuned'）
    - merge_before_training: 是否在训练前将文本合并。True表示合并后训练，False表示逐列训练后再合并。

    返回:
    - embeddings: 生成的文本向量表示
    """
    # 加载指定的模型
    model = SentenceTransformer(model_name)

    if merge_before_training:
        # 合并指定列的内容作为一个文本，并进行训练
        df['拼接字段'] = df[columns].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        embeddings = model.encode(df['拼接字段'].values.tolist(), show_progress_bar=True)
    else:
        # 对每一列分别训练后，再进行合并
        embeddings_list = []
        for col in columns:
            col_embeddings = model.encode(df[col].values.tolist(), show_progress_bar=True)
            embeddings_list.append(col_embeddings)

        # 将所有列的向量进行逐行合并（可以选择按列拼接向量）
        embeddings = np.hstack(embeddings_list)

    return embeddings



def extract_citation_edges(da, relationship,all_patents_set):
    """
    从专利数据集中提取引用关系，并生成引用边列表。

    参数:
    - da: 包含专利数据的DataFrame，必须包含 '专利序号' 和 '引用专利' 列。
    - all_patents_set: 包含所有专利序号的集合，用于检查引用专利是否在数据集中。

    返回:
    - citation_edges: 由 (citing_patent, cited_patent) 元组组成的引用边列表。
    """
    citation_edges = []
    # 遍历数据集的行，提取引用关系
    for index, row in da.iterrows():
        citing_patent = row['专利序号']
        cited_patents = row[relationship]
        # 检查 '引用专利' 是否为字符串类型，如果是字符串则分隔
        if isinstance(cited_patents, str):
            cited_patents = cited_patents.split(' | ')  # 假设被引专利以 ' | ' 分隔
        # 遍历被引专利并去除空格
        for cited_patent in map(str.strip, cited_patents):
            # 检查引用专利是否存在于数据集中
            if cited_patent and cited_patent in all_patents_set:
                citation_edges.append((citing_patent, cited_patent))  # 添加引用边
    return citation_edges


# 转为Pytorch Geometric格式
def prepare_graph_data(DG, X_concatenated, prices, centrality_types=['degree'], train_ratio=0.8, seed=46):
    """
    准备图数据用于PyTorch Geometric模型，带有节点的特征、价格标签、训练集和测试集掩码。

    参数:
    - DG: NetworkX有向图对象
    - X_concatenated: 节点的特征矩阵，确保与图中的节点顺序一致
    - prices: 每个节点的价格标签，顺序与图中的节点一致
    - centrality_types: 用于增强节点特征的中心性类型，默认使用‘degree’
    - train_ratio: 训练集的比例，默认0.8
    - seed: 用于随机打乱数据集的随机种子，默认46

    返回:
    - data_with_citations: PyTorch Geometric的Data对象，包含图的节点特征、边索引、标签、训练集和测试集掩码
    """

    # 创建节点到索引的映射
    node_mapping = {node: i for i, node in enumerate(DG.nodes())}

    # 打印节点映射的数量
    print(f"Total nodes in node_mapping: {len(node_mapping)}")
    # 打印有向图的边数量
    print(f"Number of edges in directed graph DG: {DG.number_of_edges()}")

    # 提取边列表并将节点映射为索引
    edge_list = [(node_mapping[u], node_mapping[v]) for u, v in DG.edges()]

    # 构建 PyTorch Geometric 的边索引
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 转换节点特征矩阵为 tensor，确保 X 的顺序与 node_mapping 对应
    x = torch.tensor(X_concatenated, dtype=torch.float)

    # 增强节点特征矩阵（假设 augment_with_centrality 是一个外部定义的函数）
    if centrality_types:
        x = augment_with_centrality(x, DG, centrality_types=centrality_types)

    # 转换价格标签为 tensor，并进行对数变换
    prices = [float(price) for price in prices]
    prices = torch.tensor(prices, dtype=torch.float)
    log_prices = torch.log1p(prices)

    # 设置随机种子
    np.random.seed(seed)

    # 数据划分 - 划分训练集和测试集
    num_nodes = len(node_mapping)
    num_train = int(train_ratio * num_nodes)

    # 随机打乱节点索引
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    # 划分训练集和测试集
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    # 创建 train_mask 和 test_mask
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 设置训练集和测试集的掩码
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    # 创建 PyTorch Geometric 数据对象
    data_with_citations = Data(x=x, edge_index=edge_index, y=log_prices, train_mask=train_mask, test_mask=test_mask)

    # 返回封装的数据对象
    return data_with_citations
