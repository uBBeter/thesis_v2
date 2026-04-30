import numpy as np
import torch


def _get_ranks(scores_pos: np.ndarray, scores_neg: np.ndarray) -> np.ndarray:
    rank = (scores_neg > scores_pos[:, None]).sum(axis=1)
    return rank


def recall_at_k(scores_pos: np.ndarray, scores_neg: np.ndarray, k: int) -> float:
    ranks = _get_ranks(scores_pos, scores_neg)
    return float((ranks < k).mean())


def ndcg_at_k(scores_pos: np.ndarray, scores_neg: np.ndarray, k: int) -> float:
    ranks = _get_ranks(scores_pos, scores_neg)
    hit = ranks < k
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
