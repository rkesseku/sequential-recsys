"""Second-stage re-ranker: LightGBM LambdaRank over merged EASE + SASRec candidates.

The ranker is trained on the *validation* targets (candidates built from training histories)
and evaluated on the *test* targets (candidates built from training + validation histories),
so it never sees a test label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Split
from .metrics import topk

FEATURES = [
    "ease_score", "ease_rank", "sas_score", "sas_rank", "knn_score", "knn_rank",
    "log_pop", "log_recent_pop", "log_user_len", "genre_affinity",
]


def full_ranks(scores: np.ndarray) -> np.ndarray:
    """0-based rank of every item per row (inverse of the descending argsort)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    return ranks


def merge_candidates(score_list: list[np.ndarray], n_cand: int = 100) -> np.ndarray:
    """Interleave each retriever's ranked list, de-duplicate, keep the first ``n_cand``."""
    tops = [topk(s, n_cand) for s in score_list]
    out = np.zeros((tops[0].shape[0], n_cand), dtype=np.int64)
    for u in range(out.shape[0]):
        seen: dict[int, None] = {}
        for pos in range(n_cand):
            for t in tops:
                seen.setdefault(int(t[u, pos]), None)
        out[u] = list(seen)[:n_cand]
    return out


def _genre_matrix(split: Split) -> tuple[np.ndarray, list[str]]:
    names = sorted({g for gs in split.item_genres.values() for g in gs})
    m = np.zeros((split.n_items + 1, max(len(names), 1)), dtype=np.float32)
    idx = {g: i for i, g in enumerate(names)}
    for item, gs in split.item_genres.items():
        for g in gs:
            m[item, idx[g]] = 1.0
    return m, names


@dataclass
class StageInputs:
    """Masked score matrices for one stage (val or test)."""

    ease: np.ndarray
    sas: np.ndarray
    knn: np.ndarray


def build_features(split: Split, stage: str, inp: StageInputs, cands: np.ndarray,
                   recent_window: int = 20) -> np.ndarray:
    hist = split.histories(stage)
    n_users, n_cand = cands.shape
    rows = np.repeat(np.arange(n_users), n_cand)
    flat = cands.ravel()
    counts = split.item_counts(stage).astype(np.float64)
    recent = np.zeros(split.n_items + 1)
    for h in hist:
        np.add.at(recent, np.asarray(h[-recent_window:], dtype=np.int64), 1.0)
    gm, _ = _genre_matrix(split)
    user_g = np.zeros((n_users, gm.shape[1]), dtype=np.float32)
    for u, h in enumerate(hist):
        v = gm[np.asarray(h, dtype=np.int64)].sum(axis=0)
        user_g[u] = v / max(v.sum(), 1.0)
    item_g = gm[flat]
    aff = (user_g[rows] * item_g).sum(axis=1) / np.maximum(item_g.sum(axis=1), 1.0)
    cols = [
        inp.ease[rows, flat], full_ranks(inp.ease)[rows, flat],
        inp.sas[rows, flat], full_ranks(inp.sas)[rows, flat],
        inp.knn[rows, flat], full_ranks(inp.knn)[rows, flat],
        np.log1p(counts[flat]), np.log1p(recent[flat]),
        np.log1p(np.array([len(h) for h in hist], dtype=float))[rows], aff,
    ]
    return np.column_stack(cols).astype(np.float32)


# The fixed re-ranker recipe (LightGBM names; ``n_estimators`` is the cap for early stopping).
RANKER_DEFAULTS = {
    "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50,
    "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 0.0, "n_estimators": 600,
}


def lgbm_kwargs(params: dict) -> dict:
    """Translate ``RANKER_DEFAULTS``-style names to ``lightgbm.LGBMRanker`` keyword arguments."""
    p = {**RANKER_DEFAULTS, **params}
    return {"n_estimators": int(p["n_estimators"]), "learning_rate": float(p["learning_rate"]),
            "num_leaves": int(p["num_leaves"]), "min_child_samples": int(p["min_child_samples"]),
            "subsample": float(p["bagging_fraction"]), "subsample_freq": int(p["bagging_freq"]),
            "colsample_bytree": float(p["feature_fraction"]),
            "reg_lambda": float(p["lambda_l2"])}


def train_ranker(x: np.ndarray, y: np.ndarray, n_cand: int, seed: int = 0,
                 n_estimators: int | None = None, params: dict | None = None,
                 n_jobs: int | None = None):
    """Fit LambdaRank on groups that contain the positive; 10% of users for early stopping.

    ``params`` overrides ``RANKER_DEFAULTS`` (an explicit ``n_estimators`` overrides both).
    """
    import lightgbm as lgb

    params = dict(params or {})
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    kw = lgbm_kwargs(params)
    groups = y.reshape(-1, n_cand)
    keep = np.flatnonzero(groups.sum(axis=1) > 0)
    rng = np.random.default_rng(seed)
    rng.shuffle(keep)
    n_es = max(1, len(keep) // 10)
    es, tr = np.sort(keep[:n_es]), np.sort(keep[n_es:])

    def take(users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = (users[:, None] * n_cand + np.arange(n_cand)[None, :]).ravel()
        return x[idx], y[idx]

    xtr, ytr = take(tr)
    xes, yes = take(es)
    extra = {} if n_jobs is None else {"n_jobs": n_jobs}
    model = lgb.LGBMRanker(objective="lambdarank", random_state=seed, verbose=-1, **kw, **extra)
    model.fit(xtr, ytr, group=[n_cand] * len(tr), eval_set=[(xes, yes)],
              eval_group=[[n_cand] * len(es)], eval_at=[10],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model, {"n_train_groups": int(len(tr)), "n_es_groups": int(len(es)),
                   "best_iteration": int(model.best_iteration_ or kw["n_estimators"])}


def rerank_ranks(pred: np.ndarray, cands: np.ndarray, targets: np.ndarray,
                 miss_rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Rank of the target inside the re-ordered candidate list (``miss_rank`` if absent)."""
    p = pred.reshape(cands.shape)
    order = np.argsort(-p, axis=1, kind="stable")
    reordered = np.take_along_axis(cands, order, axis=1)
    pos = reordered == targets[:, None]
    ranks = np.where(pos.any(axis=1), pos.argmax(axis=1), miss_rank)
    return ranks, reordered
