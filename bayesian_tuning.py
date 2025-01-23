import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import umap
from sklearn.preprocessing import StandardScaler
from DataLoader import generate_text_embeddings, extract_citation_edges, prepare_graph_data
import random
from DataAugmentor import DataAugmentor
from models.model_Gated import HierarchicalBidirectionalSAGE_Gated
import optuna

data_name = 'pbat_cn_low.xlsx'
features = ['公开(公告)号',"技术覆盖范围", "引证次数", "被引证次数", "文献页数", "发明人数量",
                "科学关联性", "他引率", "权利要求数量", "首权字数", "无效次数",
                "专利有效性", "诉讼次数", "审查时长", "剩余寿命", "专利年龄",
                "申请人类型",  "商业化次数", "同族国家数", "质押次数", '专利价值']
citation = True
cited = False
epochs = 3000
# Call function and move data to GPU
device = torch.device("cuda:0")

def set_seed(seed):
    torch.manual_seed(seed)  # Set CPU random seed
    torch.cuda.manual_seed(seed)  # Set current GPU random seed
    torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs if using multiple
    np.random.seed(seed)  # Set Numpy random seed
    random.seed(seed)  # Set Python random seed

    # Ensure deterministic behavior in GPU computations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Use a fixed seed to ensure reproducibility
set_seed(38)

df1 = pd.read_excel(data_name)
# Select numerical features
num_data = df1[features]

columns = ['Title (English)', 'Abstract (English)', 'Claims (English)']
embeddings = generate_text_embeddings(df1, columns, merge_before_training=False)

print('embedding Done')
print('embeddings shape :', embeddings.shape)


num_data_clean = num_data.drop(columns=['Publication Number', 'Patent Value'], axis=1).values

# Assuming df1 is the raw data containing 'Publication Number', '[Mark] Original Applicant (Patent Holder)' and 'Cited Patents'
# Example data processing - Build a cooperation network based on inventors
df1['Patent Serial Number'] = df1['Publication Number']
prices = df1['Patent Value'].tolist()

# Pre-collect all patent serial numbers to reduce duplicate checks
all_patents_set = set(df1['Patent Serial Number'].values)  # Use a set for faster lookup

citation_edges = extract_citation_edges(df1, 'Cited Patents', all_patents_set)
# Create directed graph
DG = nx.DiGraph()

# Add all patent nodes in batch
DG.add_nodes_from(all_patents_set)  # Add all patent nodes at once

if citation:
    DG.add_edges_from(citation_edges)  # Directed citation edges

# Print the number of nodes in the directed graph
print(f"Number of nodes in directed graph DG: {DG.number_of_nodes()}")

out_channels = 1  # Output dimension is 1 (predict patent value)
best_epoch = -1
# Initialize minimum test loss and evaluation metrics as very large values
min_test_loss = float('inf')
best_test_metrics = {'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'mape': float('inf'), 'r2': -float('inf')}

def objective(trial):
    centrality_types = trial.suggest_categorical('centrality_types', [['degree'], ['betweenness'], ['closeness'],
                                                                      ['degree', 'betweenness'], ['degree', 'closeness'], ['degree', 'betweenness', 'closeness'],
                                                                      ['betweenness', 'closeness']
                                                                      ])

    def process_embeddings(embeddings, pooling_type='mean', pooling_stage='none', n_components=23, random_state=42):
        def apply_pooling(embeddings, pooling_type):
            if pooling_type == 'mean':
                return np.mean(embeddings, axis=1)  # Mean pooling
            elif pooling_type == 'max':
                return np.max(embeddings, axis=1)  # Max pooling
            else:
                return embeddings  # No pooling

        # Apply pooling (if selected pre pooling or both pooling)
        if pooling_stage == 'pre' or pooling_stage == 'both':
            embeddings = apply_pooling(embeddings, pooling_type)
            if embeddings.ndim == 1:  # Reshape to make sure embeddings is 2D if collapsed into a 1D array
                embeddings = embeddings.reshape(-1, 1)

        # Create UMAP object and perform dimensionality reduction
        umap_model = umap.UMAP(n_components=n_components, random_state=random_state)
        reduced_embeddings = umap_model.fit_transform(embeddings)

        # Apply pooling after dimensionality reduction (if selected post pooling or both pooling)
        if pooling_stage == 'post' or pooling_stage == 'both':
            reduced_embeddings = apply_pooling(reduced_embeddings, pooling_type)
            if reduced_embeddings.ndim == 1:  # Reshape to make sure it's 2D if necessary
                reduced_embeddings = reduced_embeddings.reshape(-1, 1)

        return reduced_embeddings

    umap_embeddings = process_embeddings(embeddings, pooling_type='max', pooling_stage='pre')


    concatenated = np.concatenate((num_data_clean, umap_embeddings), axis=1)

    # Standardize numerical features
    scaler = StandardScaler()
    X_concatenated = scaler.fit_transform(concatenated)

    # Prepare graph data
    data_with_citations = prepare_graph_data(DG, X_concatenated, prices, centrality_types=centrality_types, train_ratio=0.8, seed=46)
    data = data_with_citations.to(device)

    # Initialize model and optimizer
    data_augmentor = DataAugmentor(
        noise_scale=0.00,  # Node feature perturbation
        drop_edge_ratio=0.0,  # Edge drop rate
        drop_node_ratio=0.0,  # Node drop rate
        graph_mixup=True,  # Graph Mixup
        subgraph_sampling=False,  # Subgraph sampling
        attribute_masking=False,  # Attribute masking
        vat=True  # Virtual Adversarial Training (VAT)
    )
    in_channels = data.num_node_features

    hidden_channels1 = 936
    hidden_channels2 = 861

    model = HierarchicalBidirectionalSAGE_Gated(in_channels, hidden_channels1, hidden_channels2, out_channels, dropout=0.271).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=7.96e-05)

    def calculate_metrics(preds, targets):
        mse = F.mse_loss(preds, targets).item()
        rmse = torch.sqrt(F.mse_loss(preds, targets)).item()  # RMSE
        mae = F.l1_loss(preds, targets).item()  # MAE
        mape = (torch.mean(torch.abs((targets - preds) / targets)) * 100).item()  # MAPE
        ss_res = torch.sum((targets - preds) ** 2).item()
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
        r2 = 1 - ss_res / ss_tot  # R²

        return rmse, mae, mape, r2

    # Training function
    def train():
        model.train()
        optimizer.zero_grad()

        if data_augmentor.vat:
            augmented_data, vat_loss = data_augmentor.augment(data, model=model)
        else:
            augmented_data = data_augmentor.augment(data)

        augmented_data = augmented_data.to(device)  # Move augmented data to GPU

        out = model(augmented_data)  # Train with augmented data
        out = out.squeeze()

        # Calculate loss only for nodes in the training set
        loss = F.mse_loss(out[augmented_data.train_mask], augmented_data.y[augmented_data.train_mask])
        loss.backward()
        optimizer.step()

        # Calculate RMSE, MAE, MAPE, and R²
        rmse, mae, mape, r2 = calculate_metrics(out[augmented_data.train_mask],
                                                augmented_data.y[augmented_data.train_mask])

        return loss.item(), rmse, mae, mape, r2

    # Testing function
    def test():
        model.eval()
        with torch.no_grad():
            out = model(data)
            out = out.squeeze()
            # Calculate loss only for nodes in the test set
            loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])

            # Calculate RMSE, MAE, MAPE, and R²
            rmse, mae, mape, r2 = calculate_metrics(out[data.test_mask], data.y[data.test_mask])

        return loss.item(), rmse, mae, mape, r2

    # Record training and testing losses for each epoch
    train_losses = []
    test_losses = []

    # Initialize minimum test loss
    min_test_loss = float('inf')
    best_test_metrics = {'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'), 'mape': float('inf'), 'r2': -float('inf')}

    for epoch in range(epochs):
        train_loss, _, _, _, _ = train()
        test_loss, test_rmse, test_mae, test_mape, test_r2 = test()

        # Record losses
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Update best test loss and evaluation metrics
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            best_test_metrics['mse'] = test_loss
            best_test_metrics['rmse'] = test_rmse
            best_test_metrics['mae'] = test_mae
            best_test_metrics['mape'] = test_mape
            best_test_metrics['r2'] = test_r2
            # Save current model and state dict
            torch.save(model, f'{data_name}_best_model.pth')  # Save entire model
            torch.save(model.state_dict(), f'{data_name}_best_model_state_dict.pth')  # Save model state dict
            torch.save(optimizer.state_dict(), f'{data_name}_best_optimizer_state_dict.pth')  # Optionally save optimizer state

    return min_test_loss

# Create an Optuna study object and perform optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# Output the best hyperparameters
print("Best hyperparameters: ", study.best_params)

# Output the best MSE and other results
print("Best Test Metrics:")
print(f'MSE: {study.best_value:.4f}')
print(f'RMSE: {best_test_metrics["rmse"]:.4f}')
print(f'MAE: {best_test_metrics["mae"]:.4f}')
print(f'MAPE: {best_test_metrics["mape"]:.2f}%')
print(f'R²: {best_test_metrics["r2"]:.4f}')
