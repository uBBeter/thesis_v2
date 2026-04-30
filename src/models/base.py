from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from ..evaluation.metrics import compute_all_metrics


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        ...

    @abstractmethod
    def get_user_embeddings(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_item_embeddings(self) -> np.ndarray:
        ...

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        U = self.get_user_embeddings()
        V = self.get_item_embeddings()
        return (U[user_ids] * V[item_ids]).sum(axis=1)

    def evaluate(
        self,
        eval_df: pd.DataFrame,
        ks: list[int] = [10, 20],
    ) -> dict[str, float]:
        user_ids = eval_df["user_idx"].values
        pos_items = eval_df["pos_item"].values
        neg_items = np.stack(eval_df["neg_items"].values)

        scores_pos = self.score(user_ids, pos_items)
        n_users, n_neg = neg_items.shape
        flat_users = np.repeat(user_ids, n_neg)
        flat_items = neg_items.flatten()
        scores_neg = self.score(flat_users, flat_items).reshape(n_users, n_neg)

        return compute_all_metrics(scores_pos, scores_neg, ks=ks)
