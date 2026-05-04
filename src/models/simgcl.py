import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .lightgcn import LightGCN
from ..data.preprocessor import Dataset
from ..utils.helpers import EarlyStopping, save_checkpoint


class SimGCL(LightGCN):
    def __init__(self, n_users: int, n_items: int, dim: int = 64,
                 n_layers: int = 3, reg: float = 1e-4,
                 ssl_temp: float = 0.2, ssl_lambda: float = 0.5,
                 noise_eps: float = 0.1):
        super().__init__(n_users, n_items, dim, n_layers, reg)
        self.ssl_temp = ssl_temp
        self.ssl_lambda = ssl_lambda
        self.noise_eps = noise_eps

    def _propagate_with_noise(self, graph: Data) -> torch.Tensor:
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
            noise = F.normalize(torch.randn_like(x), dim=1) * self.noise_eps
            x = x + noise
            all_embs.append(x)
        return torch.stack(all_embs, dim=0).mean(dim=0)

    def _info_nce(self, z1: torch.Tensor, z2: torch.Tensor,
                  indices: torch.Tensor) -> torch.Tensor:
        v1 = F.normalize(z1[indices], dim=1)
        v2 = F.normalize(z2[indices], dim=1)
        pos = (v1 * v2).sum(dim=1) / self.ssl_temp
        neg = (v1 @ v2.T) / self.ssl_temp
        loss = -pos + torch.logsumexp(neg, dim=1)
        return loss.mean()

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
        n_users = dataset.n_users

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

                embs_main = self._propagate(graph)
                z1 = self._propagate_with_noise(graph)
                z2 = self._propagate_with_noise(graph)

                bpr = self.bpr_loss(embs_main, u, p, n_t)

                u_unique = u.unique()
                p_unique = (p + n_users).unique()
                ssl = (self._info_nce(z1, z2, u_unique) +
                       self._info_nce(z1, z2, p_unique))

                loss = bpr + self.ssl_lambda * ssl
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
