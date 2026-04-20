import numpy as np
import torch


def _get_ranks(scores_pos: np.ndarray, scores_neg: np.ndarray) -> np.ndarray:
    """
    For each user, compute the rank of the positive item among pos+neg items.
    scores_pos: [n_users]
    scores_neg: [n_users, n_neg]
    Returns rank (0-based) for each user.
    """
    # rank = number of negative items scored higher than the positive
    rank = (scores_neg > scores_pos[:, None]).sum(axis=1)
    return rank  # shape [n_users]


def recall_at_k(scores_pos: np.ndarray, scores_neg: np.ndarray, k: int) -> float:
    ranks = _get_ranks(scores_pos, scores_neg)
    return float((ranks < k).mean())


def ndcg_at_k(scores_pos: np.ndarray, scores_neg: np.ndarray, k: int) -> float:
    ranks = _get_ranks(scores_pos, scores_neg)
    hit = ranks < k
    # DCG = 1 / log2(rank + 2), IDCG = 1 (best rank = 0)
    dcg = np.where(hit, 1.0 / np.log2(ranks + 2), 0.0)
    return float(dcg.mean())


def precision_at_k(scores_pos: np.ndarray, scores_neg: np.ndarray, k: int) -> float:
    ranks = _get_ranks(scores_pos, scores_neg)
    return float((ranks < k).mean() / k)


def compute_all_metrics(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
    ks: list[int] = [10, 20],
) -> dict[str, float]:
    results = {}
    for k in ks:
        results[f"Recall@{k}"] = recall_at_k(scores_pos, scores_neg, k)
        results[f"NDCG@{k}"] = ndcg_at_k(scores_pos, scores_neg, k)
        results[f"Precision@{k}"] = precision_at_k(scores_pos, scores_neg, k)
    return results
