"""Load MovieLens-1M as implicit feedback and build a chronological leave-last-out split."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Split:
    """Per-user chronological split. Item ids are 1..n_items (0 is reserved for padding).

    ``train[u]`` is every interaction except the last two, ``val[u]`` the second-to-last item
    and ``test[u]`` the last item. Validation scoring uses ``train[u]`` as history; test scoring
    uses ``train[u] + [val[u]]``.
    """

    train: list[list[int]]
    val: np.ndarray
    test: np.ndarray
    n_items: int
    item_genres: dict[int, list[str]] = field(default_factory=dict)

    @property
    def n_users(self) -> int:
        return len(self.train)

    def history(self, stage: str, u: int) -> list[int]:
        if stage == "val":
            return self.train[u]
        if stage == "test":
            return self.train[u] + [int(self.val[u])]
        raise ValueError(f"unknown stage {stage!r}")

    def histories(self, stage: str) -> list[list[int]]:
        return [self.history(stage, u) for u in range(self.n_users)]

    def targets(self, stage: str) -> np.ndarray:
        if stage not in ("val", "test"):
            raise ValueError(f"unknown stage {stage!r}")
        return self.val if stage == "val" else self.test

    def item_counts(self, stage: str = "val") -> np.ndarray:
        """Interaction counts per item id over the histories available at ``stage``."""
        c = np.zeros(self.n_items + 1, dtype=np.int64)
        for h in self.histories(stage):
            np.add.at(c, np.asarray(h, dtype=np.int64), 1)
        return c


def load_ratings(data_dir: str | Path) -> pd.DataFrame:
    """Read ``ratings.dat``; every rating is treated as one implicit interaction."""
    path = Path(data_dir) / "ratings.dat"
    return pd.read_csv(path, sep="::", engine="python", header=None,
                       names=["user", "item", "rating", "ts"])


def load_genres(data_dir: str | Path) -> dict[int, list[str]]:
    path = Path(data_dir) / "movies.dat"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="::", engine="python", header=None,
                     names=["item", "title", "genres"], encoding="latin-1")
    return {int(i): str(g).split("|") for i, g in zip(df["item"], df["genres"], strict=True)}


def build_split(df: pd.DataFrame, min_len: int = 3,
                raw_genres: dict[int, list[str]] | None = None) -> Split:
    """Sort each user's interactions by timestamp (stable, so ties keep file order)."""
    df = df.reset_index(drop=True).copy()
    df["_order"] = np.arange(len(df))
    counts = df.groupby("user")["item"].transform("size")
    df = df[counts >= min_len]
    raw_items = np.sort(df["item"].unique())
    item_map = {int(r): i + 1 for i, r in enumerate(raw_items)}
    df = df.sort_values(["user", "ts", "_order"], kind="stable")
    train, val, test = [], [], []
    for _, g in df.groupby("user", sort=True):
        seq = [item_map[int(i)] for i in g["item"].to_numpy()]
        train.append(seq[:-2])
        val.append(seq[-2])
        test.append(seq[-1])
    genres: dict[int, list[str]] = {}
    if raw_genres:
        genres = {item_map[r]: gs for r, gs in raw_genres.items() if r in item_map}
    return Split(train=train, val=np.asarray(val), test=np.asarray(test),
                 n_items=len(raw_items), item_genres=genres)


def load_movielens(data_dir: str | Path) -> Split:
    return build_split(load_ratings(data_dir), raw_genres=load_genres(data_dir))


def synthetic_log(n_users: int = 60, n_items: int = 40, seq_len: int = 12,
                  seed: int = 0) -> pd.DataFrame:
    """Tiny log with sequential structure (users mostly walk item i -> i+1), for tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        item = int(rng.integers(1, n_items + 1))
        seen: set[int] = set()
        t = 0
        while len(seen) < seq_len:
            if item not in seen:
                rows.append((u + 1, item, 5, t))
                seen.add(item)
                t += 1
            if rng.random() < 0.85:
                item = item % n_items + 1
            else:
                item = int(rng.integers(1, n_items + 1))
    return pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])


def interaction_matrix(histories: list[list[int]], n_items: int):
    """Binary user x (n_items + 1) CSR matrix; column 0 is the unused padding id."""
    from scipy.sparse import csr_matrix

    indptr = np.zeros(len(histories) + 1, dtype=np.int64)
    parts = []
    for u, h in enumerate(histories):
        uniq = np.unique(np.asarray(h, dtype=np.int64))
        parts.append(uniq)
        indptr[u + 1] = indptr[u] + len(uniq)
    idx = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return csr_matrix((np.ones(len(idx), dtype=np.float32), idx, indptr),
                      shape=(len(histories), n_items + 1))
