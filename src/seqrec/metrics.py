"""Full-catalogue ranking metrics with seen-item exclusion, plus bootstrap intervals."""

from __future__ import annotations

import numpy as np


def mask_seen(scores: np.ndarray, histories: list[list[int]]) -> np.ndarray:
    """Copy of ``scores`` with the padding column and already-seen items set to -inf."""
    s = scores.astype(np.float32, copy=True)
    s[:, 0] = -np.inf
    for r, h in enumerate(histories):
        if h:
            s[r, np.asarray(h)] = -np.inf
    return s


def target_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0-based rank of each target among all items (pessimistic on ties)."""
    t = scores[np.arange(len(targets)), targets][:, None]
    return (scores >= t).sum(axis=1) - 1


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def hit(ranks: np.ndarray, k: int) -> np.ndarray:
    return (np.asarray(ranks) < k).astype(float)


def ndcg(ranks: np.ndarray, k: int) -> np.ndarray:
    r = np.asarray(ranks, dtype=float)
    return np.where(r < k, 1.0 / np.log2(r + 2.0), 0.0)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile CI from resampling users with replacement."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 1000,
                     seed: int = 0) -> dict[str, float]:
    """CI for mean(a - b) over the same users, and the share of resamples with a <= b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p_le_zero": float((means <= 0).mean())}


def coverage(top: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's top-k list."""
    return float(len(np.unique(top)) / n_items)


def popularity_stats(top: np.ndarray, item_counts: np.ndarray) -> dict[str, float]:
    """Average recommended popularity, and the share of slots given to the top-10% head."""
    thr = np.quantile(item_counts[1:], 0.9)
    head = item_counts >= thr
    head[0] = False
    return {"arp": float(item_counts[top].mean()), "head_share": float(head[top].mean())}
