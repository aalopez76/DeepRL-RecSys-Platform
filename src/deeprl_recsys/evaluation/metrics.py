"""Recommendation metrics — CTR, nDCG, HitRate, MRR, Regret.

Stub for Phase 4 implementation.
"""

from __future__ import annotations

import numpy as np


def ctr(hits: np.ndarray) -> float:
    """Compute Click-Through Rate.

    Args:
        hits: Binary array (1=click, 0=no click).

    Returns:
        CTR value.
    """
    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))


def ndcg(relevance: np.ndarray, k: int | None = None) -> float:
    """Compute Normalized Discounted Cumulative Gain.

    Args:
        relevance: Array of relevance scores.
        k: Truncation point (None = full list).

    Returns:
        nDCG@k value.
    """
    if len(relevance) == 0:
        return 0.0
    if k is not None:
        relevance = relevance[:k]
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
    ideal = np.sort(relevance)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def hit_rate(hits: np.ndarray) -> float:
    """Compute Hit Rate (proportion of sessions with at least one hit).

    Args:
        hits: Binary array per session.

    Returns:
        Hit rate value.
    """
    if len(hits) == 0:
        return 0.0
    return float(np.any(hits))


def mrr(ranks: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        ranks: Array of ranks (1-indexed) of the first relevant item.

    Returns:
        MRR value.
    """
    if len(ranks) == 0:
        return 0.0
    return float(np.mean(1.0 / ranks[ranks > 0]))
