"""Non-sequential retrievers: popularity, item-kNN (cosine) and EASE.

Every model exposes ``fit(histories, n_items)`` and ``score(histories) -> (n, n_items + 1)``.
Scores are raw (unmasked); the caller removes the padding column and seen items.
"""

from __future__ import annotations

import numpy as np

from .data import interaction_matrix


class Popularity:
    name = "Popularity"

    def fit(self, histories: list[list[int]], n_items: int) -> Popularity:
        c = np.zeros(n_items + 1, dtype=np.float32)
        for h in histories:
            np.add.at(c, np.asarray(h, dtype=np.int64), 1.0)
        self.counts_ = c
        return self

    def score(self, histories: list[list[int]]) -> np.ndarray:
        return np.tile(self.counts_, (len(histories), 1))


class ItemKNN:
    """Item-item cosine similarity on the binary user-item matrix, top-k neighbours per item."""

    name = "ItemKNN"

    def __init__(self, k: int = 100, shrink: float = 10.0):
        self.k = k
        self.shrink = shrink

    def fit(self, histories: list[list[int]], n_items: int) -> ItemKNN:
        x = interaction_matrix(histories, n_items)
        co = (x.T @ x).toarray().astype(np.float32)
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.shrink + 1e-8)
        np.fill_diagonal(sim, 0.0)
        if self.k < sim.shape[0]:
            drop = np.argpartition(-sim, kth=self.k, axis=0)[self.k:, :]
            np.put_along_axis(sim, drop, 0.0, axis=0)
        self.sim_ = sim
        self.n_items = n_items
        return self

    def score(self, histories: list[list[int]]) -> np.ndarray:
        x = interaction_matrix(histories, self.n_items)
        return np.asarray(x @ self.sim_, dtype=np.float32)


class EASE:
    """Embarrassingly Shallow Autoencoder (Steck, 2019): closed-form item-item weights."""

    name = "EASE"

    def __init__(self, lam: float = 500.0):
        self.lam = lam

    def fit(self, histories: list[list[int]], n_items: int) -> EASE:
        x = interaction_matrix(histories, n_items)
        g = (x.T @ x).toarray().astype(np.float64)
        g[np.diag_indices_from(g)] += self.lam
        p = np.linalg.inv(g)
        b = -p / np.diag(p)
        np.fill_diagonal(b, 0.0)
        self.b_ = b.astype(np.float32)
        self.n_items = n_items
        return self

    def score(self, histories: list[list[int]]) -> np.ndarray:
        x = interaction_matrix(histories, self.n_items)
        return np.asarray(x @ self.b_, dtype=np.float32)
