import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

def build_unified_err_graph(csv_path, feature_columns, out_path='./', k_neighbors=8):   
    df = pd.read_csv(csv_path).copy()
    
    X = df[feature_columns].values.astype(float)
    Y = df[["error","cpu_time"]].values.astype(float)
    
    # --- Normalize ---
    scaler_x = StandardScaler()
    X_norm = scaler_x.fit_transform(X)
    
    scaler_y = StandardScaler()
    Y_norm = scaler_y.fit_transform(Y)
    
    n_samples = X_norm.shape[0]
    k = min(k_neighbors, max(1, n_samples - 1))
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(X_norm)
    distances, indices = nbrs.kneighbors(X_norm)
    
    edge_list = [] 
    for i in range(n_samples): 
        neigh = indices[i] 
        for j in neigh: 
            if j == i: 
                continue 
            edge_list.append([i, j])
    
    # convert to torch edge_index (2 x E) tensor = [[source nodes],[target nodes]]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    x = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(Y_norm, dtype=torch.float32)
    
    orig_targets = torch.tensor(Y, dtype=torch.float32)
    
    data = Data(x=x, edge_index=edge_index, y=y_t)
    data.orig_targets = orig_targets
    data.df_index = df.index.values
    data.raw_df = df 
    
    # Save everything
    torch.save({
        "data": data,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_cols": feature_columns,
        "raw_df": df
    }, out_path)
    
    print(f"Built graph with {n_samples} nodes, {edge_index.shape[1]} edges. Saved to {out_path}")
    print("Feature columns:", feature_columns)
    return data, scaler_x, scaler_y

def load_graph(graph_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(graph_path, map_location=device, weights_only=False)
    data = checkpoint['data'].to(device)
    scaler_x = checkpoint['scaler_x']
    scaler_y = checkpoint['scaler_y']
    return data, scaler_x, scaler_y

def mask_data(data, split=0.8):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)  # shuffle node indices
    
    train_size = int(split * num_nodes)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x

import torch

def load_trained_gnn_model(model_path, graph_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load graph + scalers
    data, scaler_x, scaler_y = load_graph(graph_path)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Build model
    model_loaded = GCN(
        in_channels=data.x.shape[1],
        hidden_channels=64,
        out_channels=data.y.shape[1]
    ).to(device)

    model_loaded.load_state_dict(checkpoint["model_state"])
    model_loaded.eval()

    print(f"Loaded trained GNN model from: {model_path}")
    print(f"Loaded graph from: {graph_path}")

    return model_loaded, data, scaler_x, scaler_y

# if __name__ == "__main__":
#     print(data)