import torch
import scipy.sparse as sp
import numpy as np
from torch_geometric.data import Data
from .preprocessor import Dataset


def build_graph(dataset: Dataset) -> Data:
    n_users = dataset.n_users
    n_items = dataset.n_items
    n_nodes = n_users + n_items

    train = dataset.train
    user_ids = train["user_idx"].values
    item_ids = train["item_idx"].values + n_users

    rows = np.concatenate([user_ids, item_ids])
    cols = np.concatenate([item_ids, user_ids])
    vals = np.ones(len(rows), dtype=np.float32)

    adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))

    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    d_inv_sqrt[nonzero] = degree[nonzero] ** -0.5
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
