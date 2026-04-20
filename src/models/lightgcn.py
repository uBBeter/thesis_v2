"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
He et al., 2020 (https://arxiv.org/abs/2002.02126)

Key insight: feature transformation and non-linear activation in NGCF are harmful for CF.
Remove both. Only keep neighborhood aggregation.
Final embedding = mean of all layer embeddings (instead of concatenation).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .base import BaseRecommender
from ..data.preprocessor import Dataset
from ..data.graph import build_graph
from ..utils.helpers import EarlyStopping, save_checkpoint


class LightGCN(nn.Module, BaseRecommender):
    def __init__(self, n_users: int, n_items: int, dim: int = 64,
                 n_layers: int = 3, reg: float = 1e-4):
        nn.Module.__init__(self)
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.reg = reg

        self.embedding = nn.Embedding(n_users + n_items, dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def _propagate(self, graph: Data) -> torch.Tensor:
        """Run L layers of light graph convolution, return mean of all layers."""
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight
        x = self.embedding.weight
        all_embs = [x]
        row, col = edge_index
        for _ in range(self.n_layers):
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)),
                             edge_weight.unsqueeze(1) * x[row])
            x = agg
            all_embs.append(x)
        # Mean pooling across layers (including layer 0)
        return torch.stack(all_embs, dim=0).mean(dim=0)

    def forward(self, graph: Data) -> torch.Tensor:
        return self._propagate(graph)

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
                n_t = torch.tensor(neg_items, device=device)

                embs = self.forward(graph)
                loss = self.bpr_loss(embs, u, p, n_t)
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
