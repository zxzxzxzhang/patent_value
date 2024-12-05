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

from sentence_transformers import SentenceTransformer
import numpy as np
from centrality import augment_with_centrality
import torch
from torch_geometric.data import Data
import umap

# embedding
def generate_text_embeddings(df, columns, model_name='distilbert_finetuned', merge_before_training=True):
    """
    Generate text embeddings for specified columns, supporting either merging columns before training
    or training each column separately and then combining the results.

    Parameters:
    - df: Input DataFrame
    - columns: List of column names to process
    - model_name: Name of the pre-trained model to use (default: 'distilbert_finetuned')
    - merge_before_training: Whether to merge text before training.
                             True means merging columns into one text before training,
                             False means training each column separately and then combining the embeddings.

    Returns:
    - embeddings: Generated text embeddings
    """
    # Load the specified model
    model = SentenceTransformer(model_name)

    if merge_before_training:

        df['拼接字段'] = df[columns].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        embeddings = model.encode(df['拼接字段'].values.tolist(), show_progress_bar=True)
    else:

        embeddings_list = []
        for col in columns:
            col_embeddings = model.encode(df[col].values.tolist(), show_progress_bar=True)
            embeddings_list.append(col_embeddings)


        embeddings = np.hstack(embeddings_list)

    return embeddings



def extract_citation_edges(da, relationship,all_patents_set):
    """
    Extract citation relationships from a patent dataset and generate a list of citation edges.

    Parameters:
    - da: A DataFrame containing patent data, which must include the columns 'Patent Number' and 'Cited Patent'.
    - all_patents_set: A set containing all patent numbers, used to check if the cited patents are within the dataset.

    Returns:
    - citation_edges: A list of citation edges represented as (citing_patent, cited_patent) tuples.
    """

    citation_edges = []

    for index, row in da.iterrows():
        citing_patent = row['专利序号']
        cited_patents = row[relationship]

        if isinstance(cited_patents, str):
            cited_patents = cited_patents.split(' | ')

        for cited_patent in map(str.strip, cited_patents):

            if cited_patent and cited_patent in all_patents_set:
                citation_edges.append((citing_patent, cited_patent))
    return citation_edges


# To Pytorch Geometric
def prepare_graph_data(DG, X_concatenated, prices, centrality_types=['degree'], train_ratio=0.8, seed=46):
    """
    Prepare graph data for PyTorch Geometric models, including node features, price labels,
    training and testing masks.

    Parameters:
    - DG: NetworkX directed graph object
    - X_concatenated: Feature matrix for nodes, ensuring it aligns with the order of nodes in the graph
    - prices: Price labels for each node, aligned with the order of nodes in the graph
    - centrality_types: Types of centrality measures to enhance node features, default is 'degree'
    - train_ratio: Ratio of training data, default is 0.8
    - seed: Random seed for shuffling the dataset, default is 46

    Returns:
    - data_with_citations: PyTorch Geometric Data object containing node features, edge indices,
      labels, and training/testing masks for the graph
    """

    node_mapping = {node: i for i, node in enumerate(DG.nodes())}


    print(f"Total nodes in node_mapping: {len(node_mapping)}")
    print(f"Number of edges in directed graph DG: {DG.number_of_edges()}")


    edge_list = [(node_mapping[u], node_mapping[v]) for u, v in DG.edges()]


    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()


    x = torch.tensor(X_concatenated, dtype=torch.float)


    if centrality_types:
        x = augment_with_centrality(x, DG, centrality_types=centrality_types)


    prices = [float(price) for price in prices]
    prices = torch.tensor(prices, dtype=torch.float)
    log_prices = torch.log1p(prices)

    # seed
    np.random.seed(seed)

    # split
    num_nodes = len(node_mapping)
    num_train = int(train_ratio * num_nodes)

    # shuffle
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    data_with_citations = Data(x=x, edge_index=edge_index, y=log_prices, train_mask=train_mask, test_mask=test_mask)

    return data_with_citations

def process_embeddings(embeddings, pooling_type='mean', pooling_stage='none', n_components=23, random_state=42):
    def apply_pooling(embeddings, pooling_type):
        if pooling_type == 'mean':
            return np.mean(embeddings, axis=1)  # Mean pooling
        elif pooling_type == 'max':
            return np.max(embeddings, axis=1)   # Max pooling
        else:
            return embeddings  # 不进行 pooling

    # pooling
    if pooling_stage == 'pre' or pooling_stage == 'both':
        embeddings = apply_pooling(embeddings, pooling_type)
        if embeddings.ndim == 1:  # Reshape
            embeddings = embeddings.reshape(-1, 1)

    # UMAP
    umap_model = umap.UMAP(n_components=n_components, random_state=random_state)
    reduced_embeddings = umap_model.fit_transform(embeddings)

    if pooling_stage == 'post' or pooling_stage == 'both':
        reduced_embeddings = apply_pooling(reduced_embeddings, pooling_type)
        if reduced_embeddings.ndim == 1:  # Reshape
            reduced_embeddings = reduced_embeddings.reshape(-1, 1)

    return reduced_embeddings

