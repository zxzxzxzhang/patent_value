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
from DataLoader import generate_text_embeddings, extract_citation_edges, prepare_graph_data, process_embeddings
import random
from DataAugmentor import DataAugmentor
from models.model_gcn import GCN
from models.model_gat import GAT
from models.model_multi_sage import Multi_SAGE_Gated

data_name = 'data.xlsx'
#features = ['patent number','patent value']
citation = True
cited = False
umap_dimension = 20
epochs = 5000
centrality_types= ['degree']
noise_scale = 0.0
drop_edge_ratio = 0.0
drop_node_ratio = 0.0
graph_mixup = True
subgraph_sampling = False
attribute_masking = False
vat = True

dropout = 0.2
hidden_channels1 = 1024
hidden_channels2 = 896
lr = 0.001
graph_model = GAT

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(38)

df1 = pd.read_excel(data_name)

num_data = df1[features]

columns = ['title', 'abstract', 'claim']
embeddings = generate_text_embeddings(df1, columns, merge_before_training=False)

print('embedding Done')
print('embeddings shape :',embeddings.shape)
umap_embeddings = process_embeddings(embeddings, pooling_type='none', pooling_stage='none')
print('UMAP Done')
print('umap_embeddings shape:', umap_embeddings.shape)
num_data_clean = num_data.drop(columns=['patent number','patent value'], axis=1).values

concatenated = np.concatenate((num_data_clean, umap_embeddings), axis=1)
scaler = StandardScaler()
X_concatenated = scaler.fit_transform(concatenated)
print('X_concatenated.shape :',X_concatenated.shape)


df1['Patent serial number'] = df1['patent number']
prices = df1['patent value'].tolist()

all_patents_set = set(df1['Patent serial number'].values)

citation_edges = extract_citation_edges(df1, '引用专利', all_patents_set)

DG = nx.DiGraph()

# add nodes
DG.add_nodes_from(all_patents_set)

if citation:
    DG.add_edges_from(citation_edges)


print(f"Number of nodes in directed graph DG: {DG.number_of_nodes()}")

data_with_citations = prepare_graph_data(DG, X_concatenated, prices, centrality_types=centrality_types, train_ratio=0.8, seed=46)
device = torch.device("cuda")
data = data_with_citations.to(device)

data_augmentor = DataAugmentor(
    noise_scale=noise_scale,           # Node feature perturbation
    drop_edge_ratio=drop_edge_ratio,   # Edge dropout ratio
    drop_node_ratio=drop_node_ratio,   # Node dropout ratio

    graph_mixup=graph_mixup,           # Graph Mixup
    subgraph_sampling=subgraph_sampling, # Subgraph sampling
    attribute_masking=attribute_masking, # Attribute masking
    vat=vat                            # Virtual Adversarial Training (VAT)
)

# calculate_metrics RMSE MAE MAPE
def calculate_metrics(preds, targets):
    mse = F.mse_loss(preds, targets).item()
    rmse = torch.sqrt(F.mse_loss(preds, targets)).item()  # RMSE
    mae = F.l1_loss(preds, targets).item()  # MAE
    mape = (torch.mean(torch.abs((targets - preds) / targets)) * 100).item()  # MAPE
    ss_res = torch.sum((targets - preds) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1 - ss_res / ss_tot

    return rmse, mae, mape, r2

def train():
    model.train()
    optimizer.zero_grad()

    if data_augmentor.vat:
        augmented_data, vat_loss = data_augmentor.augment(data, model=model)
    else:
        augmented_data = data_augmentor.augment(data)

    augmented_data = augmented_data.to(device)

    out = model(augmented_data)
    out = out.squeeze()

    loss = F.mse_loss(out[augmented_data.train_mask], augmented_data.y[augmented_data.train_mask])
    loss.backward()
    optimizer.step()

    rmse, mae, mape, r2 = calculate_metrics(out[augmented_data.train_mask], augmented_data.y[augmented_data.train_mask])

    return loss.item(), rmse, mae, mape, r2

# test
def test():
    model.eval()
    with torch.no_grad():
        out = model(data)
        out = out.squeeze()
        loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])

        rmse, mae, mape, r2 = calculate_metrics(out[data.test_mask], data.y[data.test_mask])

    return loss.item(), rmse, mae, mape, r2

in_channels = data.num_node_features
out_channels = 1
best_epoch = -1

model = graph_model(in_channels, hidden_channels1, hidden_channels2,out_channels, dropout=dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)

train_losses = []
test_losses = []
train_metrics = []
test_metrics = []


min_test_loss = float('inf')
best_test_metrics = {'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'mape': float('inf'), 'r2': -float('inf')}


for epoch in range(epochs):
    train_loss, train_rmse, train_mae, train_mape, train_r2 = train()
    test_loss, test_rmse, test_mae, test_mape, test_r2 = test()


    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_metrics.append((train_rmse, train_mae, train_mape, train_r2))
    test_metrics.append((test_rmse, test_mae, test_mape, test_r2))


    if test_loss < min_test_loss:
        min_test_loss = test_loss
        best_epoch = epoch
        best_test_metrics['rmse'] = test_rmse
        best_test_metrics['mae'] = test_mae
        best_test_metrics['mape'] = test_mape
        best_test_metrics['r2'] = test_r2


    if epoch % 100 == 0:
        print(f'Epoch: {epoch:04d}')
        print(f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
        print(f'Train Metrics - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.2f}%, R²: {train_r2:.4f}')
        print(f'Test  Metrics - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}')
        print('-' * 80)


print("Best Test Metrics:")
print(f'MSE: {min_test_loss:.4f}(achieved at epoch {best_epoch})')
print(f'RMSE: {best_test_metrics["rmse"]:.4f}')
print(f'MAE: {best_test_metrics["mae"]:.4f}')
print(f'MAPE: {best_test_metrics["mape"]:.2f}%')
print(f'R²: {best_test_metrics["r2"]:.4f}')


plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_losses, label="Train Loss (MSE)", color="blue")
plt.plot(range(epochs), test_losses, label="Test Loss (MSE)", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Train and Test Loss Over Epochs")
plt.legend()
plt.show()

