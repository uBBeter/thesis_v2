import torch
import scipy.sparse as sp
import numpy as np
from torch_geometric.data import Data
from .preprocessor import Dataset


def build_graph(dataset: Dataset) -> Data:
    """
    Build a symmetric bipartite user-item graph for GNN propagation.

    Node IDs: users 0..n_users-1, items n_users..n_users+n_items-1
    Edges: undirected (user->item and item->user)
    Edge weights: symmetric normalized as in LightGCN (D^{-1/2} A D^{-1/2})

    Returns a PyG Data object with:
      - edge_index: [2, 2*n_edges]
      - edge_weight: [2*n_edges]
      - n_users, n_items
    """
    n_users = dataset.n_users
    n_items = dataset.n_items
    n_nodes = n_users + n_items

    train = dataset.train
    user_ids = train["user_idx"].values
    item_ids = train["item_idx"].values + n_users  # offset items

    # Build adjacency: user->item and item->user
    rows = np.concatenate([user_ids, item_ids])
    cols = np.concatenate([item_ids, user_ids])
    vals = np.ones(len(rows), dtype=np.float32)

    adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    norm_adj = norm_adj.tocoo()

    edge_index = torch.tensor(
        np.stack([norm_adj.row, norm_adj.col], axis=0), dtype=torch.long
    )
    edge_weight = torch.tensor(norm_adj.data, dtype=torch.float32)

    data = Data(edge_index=edge_index, edge_weight=edge_weight)
    data.n_users = n_users
    data.n_items = n_items
    data.n_nodes = n_nodes
    return data
