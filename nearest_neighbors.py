# nearest_neighbors.py

from __future__ import annotations
import numpy as np
from typing import List, Tuple


def pairwise_distances(
    X: np.ndarray,
    query: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute distances between query vector and all rows in X.
    """
    if metric == "euclidean":
        return np.linalg.norm(X - query, axis=1)

    elif metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        qn = query / (np.linalg.norm(query) + 1e-12)
        return 1.0 - (Xn @ qn)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def get_k_nearest(
    X: np.ndarray,
    index: int,
    k: int,
    metric: str,
) -> List[int]:
    """
    Return indices of k nearest neighbours (excluding self).
    """
    dists = pairwise_distances(X, X[index], metric)
    order = np.argsort(dists)
    order = order[order != index]  # exclude self
    return order[:k].tolist()


def get_k_nearest_anchors(
    X: np.ndarray,
    anchor_X: np.ndarray,
    index: int,
    k: int,
    metric: str,
) -> List[int]:
    """
    Nearest anchors to a given image feature.
    """
    q = X[index]

    if metric == "euclidean":
        dists = np.linalg.norm(anchor_X - q, axis=1)
    elif metric == "cosine":
        An = anchor_X / (np.linalg.norm(anchor_X, axis=1, keepdims=True) + 1e-12)
        qn = q / (np.linalg.norm(q) + 1e-12)
        dists = 1.0 - (An @ qn)
    else:
        raise ValueError(metric)

    return np.argsort(dists)[:k].tolist()
