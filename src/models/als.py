import numpy as np
import scipy.sparse as sp
import implicit
from .base import BaseRecommender
from ..data.preprocessor import Dataset


class ALSRecommender(BaseRecommender):
    """
    ALS (Alternating Least Squares) via the implicit library.
    Non-graph baseline — pure matrix factorization on the user-item interaction matrix.
    """

    def __init__(self, factors: int = 64, iterations: int = 50, regularization: float = 0.01,
                 use_gpu: bool = False):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            use_gpu=use_gpu,
        )
        self._user_emb: np.ndarray | None = None
        self._item_emb: np.ndarray | None = None

    def fit(self, dataset: Dataset, **kwargs):
        n_users, n_items = dataset.n_users, dataset.n_items
        train = dataset.train

        # Build sparse item-user matrix (implicit expects item x user)
        rows = train["item_idx"].values
        cols = train["user_idx"].values
        vals = np.ones(len(rows), dtype=np.float32)
        item_user = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_users))

        self.model.fit(item_user)
        self._user_emb = np.array(self.model.user_factors)
        self._item_emb = np.array(self.model.item_factors)

    def get_user_embeddings(self) -> np.ndarray:
        return self._user_emb

    def get_item_embeddings(self) -> np.ndarray:
        return self._item_emb
