from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from ..evaluation.metrics import compute_all_metrics


class BaseRecommender(ABC):
    """Abstract base for all recommender models."""

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Train the model."""
        ...

    @abstractmethod
    def get_user_embeddings(self) -> np.ndarray:
        """Return user embedding matrix [n_users, dim]."""
        ...

    @abstractmethod
    def get_item_embeddings(self) -> np.ndarray:
        """Return item embedding matrix [n_items, dim]."""
        ...

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Compute dot-product scores for (user, item) pairs.
        user_ids: [n], item_ids: [n]  -> scores: [n]
        """
        U = self.get_user_embeddings()
        V = self.get_item_embeddings()
        return (U[user_ids] * V[item_ids]).sum(axis=1)

    def evaluate(
        self,
        eval_df: pd.DataFrame,
        ks: list[int] = [10, 20],
    ) -> dict[str, float]:
        """
        Evaluate on a split DataFrame with columns:
          user_idx, pos_item, neg_items (list of 99 item indices)
        """
        user_ids = eval_df["user_idx"].values
        pos_items = eval_df["pos_item"].values
        neg_items = np.stack(eval_df["neg_items"].values)  # [n_users, n_neg]

        scores_pos = self.score(user_ids, pos_items)
        # Score each negative: vectorize by flattening then reshaping
        n_users, n_neg = neg_items.shape
        flat_users = np.repeat(user_ids, n_neg)
        flat_items = neg_items.flatten()
        scores_neg = self.score(flat_users, flat_items).reshape(n_users, n_neg)

        return compute_all_metrics(scores_pos, scores_neg, ks=ks)
