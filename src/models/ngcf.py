import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

from .base import BaseRecommender
from ..data.preprocessor import Dataset
from ..data.graph import build_graph
from ..utils.helpers import EarlyStopping, save_checkpoint


class NGCFConv(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.W1 = nn.Linear(dim, dim, bias=True)
        self.W2 = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)),
                         edge_weight.unsqueeze(1) * x[row])
        out = self.act(self.W1(agg + x) + self.W2(agg * x))
        return self.dropout(out)


class NGCF(nn.Module, BaseRecommender):
    def __init__(self, n_users: int, n_items: int, dim: int = 64,
                 n_layers: int = 3, dropout: float = 0.1, reg: float = 1e-4):
        nn.Module.__init__(self)
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.reg = reg

        self.embedding = nn.Embedding(n_users + n_items, dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.layers = nn.ModuleList([NGCFConv(dim, dropout) for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, graph: Data) -> torch.Tensor:
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight
        x = self.embedding.weight
        all_embs = [x]
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            all_embs.append(x)
        return torch.cat(all_embs, dim=1)

    def bpr_loss(self, embs: torch.Tensor, users: torch.Tensor,
                 pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        u = embs[users]
        p = embs[self.n_users + pos]
        n = embs[self.n_users + neg]
        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg_loss = self.reg * (
            self.embedding(users).norm(2).pow(2) +
            self.embedding(self.n_users + pos).norm(2).pow(2) +
            self.embedding(self.n_users + neg).norm(2).pow(2)
        ) / len(users)
        return bpr + reg_loss

    def fit(self, dataset: Dataset, graph: Data, n_epochs: int = 100, lr: float = 1e-3,
            batch_size: int = 2048, device: str = "cpu", checkpoint_path: str = None,
            val_df=None, patience: int = 10, **kwargs):
        device = torch.device(device)
        self.to(device)
        graph = graph.to(device)
        self._graph = graph

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train = dataset.train
        users_arr = train["user_idx"].values
        items_arr = train["item_idx"].values
        n_items = dataset.n_items

        stopper = EarlyStopping(patience=patience)
        rng = np.random.default_rng(42)

        for epoch in range(1, n_epochs + 1):
            self.train()
            perm = rng.permutation(len(users_arr))
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(perm), batch_size):
                idx = perm[start: start + batch_size]
                u = torch.tensor(users_arr[idx], device=device)
                p = torch.tensor(items_arr[idx], device=device)
                neg_items = rng.integers(0, n_items, size=len(idx))
                n = torch.tensor(neg_items, device=device)

                embs = self.forward(graph)
                loss = self.bpr_loss(embs, u, p, n)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if val_df is not None and epoch % 5 == 0:
                self.eval()
                with torch.no_grad():
                    metrics = self.evaluate(val_df)
                recall20 = metrics["Recall@20"]
                is_best = stopper.step(recall20)
                print(f"Epoch {epoch:3d} | loss={total_loss/n_batches:.4f} | "
                      f"Recall@20={recall20:.4f}")
                if is_best and checkpoint_path:
                    save_checkpoint({"epoch": epoch, "state_dict": self.state_dict(),
                                     "metrics": metrics}, checkpoint_path)
                if stopper.should_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | loss={total_loss/n_batches:.4f}")

        self._graph = graph
        return self

    @torch.no_grad()
    def get_user_embeddings(self) -> np.ndarray:
        self.eval()
        embs = self.forward(self._graph)
        return embs[:self.n_users].cpu().numpy()

    @torch.no_grad()
    def get_item_embeddings(self) -> np.ndarray:
        self.eval()
        embs = self.forward(self._graph)
        return embs[self.n_users:].cpu().numpy()
