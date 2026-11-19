"""End-to-end experiment: tune, fit and evaluate every model, then write metrics.json."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import Split
from .metrics import (
    bootstrap_ci,
    coverage,
    hit,
    mask_seen,
    ndcg,
    paired_bootstrap,
    popularity_stats,
    target_ranks,
    topk,
)
from .models import EASE, ItemKNN, Popularity
from .rerank import (
    FEATURES,
    StageInputs,
    build_features,
    merge_candidates,
    rerank_ranks,
    train_ranker,
)
from .sasrec import SASRec, SASRecConfig

EASE_GRID = (100.0, 250.0, 500.0, 1000.0, 2000.0)
KNN_GRID = (25, 50, 100, 200)
RECALL_KS = (1, 5, 10, 20, 50, 100)


def stage_scores(model, split: Split, stage: str) -> np.ndarray:
    hist = split.histories(stage)
    return mask_seen(model.score(hist), hist)


def val_ndcg(model, split: Split, k: int = 10) -> float:
    s = stage_scores(model, split, "val")
    return float(ndcg(target_ranks(s, split.val), k).mean())


def summarize(ranks: np.ndarray, top10: np.ndarray, split: Split, n_boot: int,
              seed: int = 0) -> dict:
    counts = split.item_counts("val")
    out: dict = {}
    for key, vals in (("hr10", hit(ranks, 10)), ("ndcg10", ndcg(ranks, 10))):
        m, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        out[key] = {"mean": m, "lo": lo, "hi": hi}
    out["recall50"] = float(hit(ranks, 50).mean())
    out["recall100"] = float(hit(ranks, 100).mean())
    out["coverage10"] = coverage(top10, split.n_items)
    out.update({f"{k}10": v for k, v in popularity_stats(top10, counts).items()})
    out["recall_curve"] = {str(k): float(hit(ranks, k).mean()) for k in RECALL_KS}
    return out


def sasrec_for_seed(split: Split, cfg: SASRecConfig, cache_dir: Path | None,
                    verbose: bool = False) -> SASRec:
    """Load a cached checkpoint for this seed if present, otherwise train and cache it."""
    import torch

    path = cache_dir / f"sasrec_seed{cfg.seed}.pt" if cache_dir else None
    model = SASRec(cfg)
    if path is not None and path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.cfg = SASRecConfig(**ck["cfg"])
        from .sasrec import SASRecNet

        model.n_items = split.n_items
        model.net = SASRecNet(split.n_items, model.cfg)
        model.net.load_state_dict(ck["state"])
        model.history_ = ck["history"]
        model.best_val_ = ck["best_val"]
        model.train_seconds_ = ck["train_seconds"]
        return model
    model.fit(split.train, split.n_items, val_fn=lambda m: val_ndcg(m, split), verbose=verbose)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(model.cfg), "state": model.net.state_dict(),
                    "history": model.history_, "best_val": model.best_val_,
                    "train_seconds": model.train_seconds_}, path)
    return model


def run_pipeline(split: Split, out_dir: str | Path, sasrec_cfg: SASRecConfig | None = None,
                 seeds: tuple[int, ...] = (0, 1, 2), cache_dir: str | Path | None = None,
                 n_boot: int = 1000, n_cand: int = 100, ease_grid=EASE_GRID,
                 knn_grid=KNN_GRID, verbose: bool = False) -> dict:
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else None
    base_cfg = sasrec_cfg or SASRecConfig()
    log = print if verbose else (lambda *a, **k: None)

    n_inter = sum(len(h) for h in split.train) + 2 * split.n_users
    res: dict = {
        "dataset": {"n_users": split.n_users, "n_items": split.n_items,
                    "n_interactions": int(n_inter),
                    "mean_train_len": float(np.mean([len(h) for h in split.train]))},
        "protocol": ("implicit feedback; per-user chronological leave-last-out (last item = test, "
                     "second-to-last = validation); full-catalogue ranking with seen items "
                     "removed; models fit on training interactions only; test history = "
                     "training + validation item"),
        "tuning": {}, "models": {},
    }

    # --- tune and fit the non-sequential retrievers on training interactions -------------
    knn_res = {str(k): val_ndcg(ItemKNN(k=k).fit(split.train, split.n_items), split)
               for k in knn_grid}
    best_k = int(max(knn_res, key=knn_res.get))
    ease_res = {str(lam): val_ndcg(EASE(lam).fit(split.train, split.n_items), split)
                for lam in ease_grid}
    best_lam = float(max(ease_res, key=ease_res.get))
    res["tuning"] = {"itemknn_k_val_ndcg10": knn_res, "itemknn_best_k": best_k,
                     "ease_lambda_val_ndcg10": ease_res, "ease_best_lambda": best_lam}
    log("tuning", res["tuning"])

    scores: dict[str, dict[str, np.ndarray]] = {}
    for model in (Popularity(), ItemKNN(k=best_k), EASE(best_lam)):
        model.fit(split.train, split.n_items)
        scores[model.name] = {st: stage_scores(model, split, st) for st in ("val", "test")}

    # --- SASRec, several seeds -----------------------------------------------------------
    seed_rows, curves = [], {}
    for seed in seeds:
        cfg = SASRecConfig(**{**asdict(base_cfg), "seed": seed})
        m = sasrec_for_seed(split, cfg, cache, verbose=verbose)
        test_s = stage_scores(m, split, "test")
        r = target_ranks(test_s, split.test)
        seed_rows.append({"seed": seed, "best_val_ndcg10": float(m.best_val_),
                          "test_ndcg10": float(ndcg(r, 10).mean()),
                          "test_hr10": float(hit(r, 10).mean()),
                          "epochs_run": len(m.history_), "train_seconds": float(m.train_seconds_)})
        curves[str(seed)] = [{k: v for k, v in h.items() if k in ("epoch", "loss", "val_ndcg10")}
                             for h in m.history_]
        if seed == seeds[0]:
            scores["SASRec"] = {"val": stage_scores(m, split, "val"), "test": test_s}
        log("sasrec", seed_rows[-1])
    nd = np.array([r["test_ndcg10"] for r in seed_rows])
    hr = np.array([r["test_hr10"] for r in seed_rows])
    res["sasrec_config"] = {k: v for k, v in asdict(base_cfg).items() if k != "seed"}
    res["sasrec_seeds"] = seed_rows
    res["sasrec_seed_summary"] = {
        "ndcg10_mean": float(nd.mean()), "ndcg10_sd": float(nd.std(ddof=1)) if len(nd) > 1 else 0.0,
        "hr10_mean": float(hr.mean()), "hr10_sd": float(hr.std(ddof=1)) if len(hr) > 1 else 0.0,
        "reported_seed": int(seeds[0])}
    res["sasrec_curves"] = curves

    per_user: dict[str, np.ndarray] = {}
    for name, st in scores.items():
        r = target_ranks(st["test"], split.test)
        per_user[name] = ndcg(r, 10)
        res["models"][name] = summarize(r, topk(st["test"], 10), split, n_boot)

    # --- re-ranker -------------------------------------------------------------------------
    inputs = {st: StageInputs(ease=scores["EASE"][st], sas=scores["SASRec"][st],
                              knn=scores["ItemKNN"][st]) for st in ("val", "test")}
    cands = {st: merge_candidates([inputs[st].ease, inputs[st].sas], n_cand)
             for st in ("val", "test")}
    xv = build_features(split, "val", inputs["val"], cands["val"])
    yv = (cands["val"] == split.val[:, None]).ravel().astype(int)
    ranker, rinfo = train_ranker(xv, yv, n_cand)
    xt = build_features(split, "test", inputs["test"], cands["test"])
    rr, reordered = rerank_ranks(ranker.predict(xt), cands["test"], split.test, split.n_items)
    per_user["Reranker"] = ndcg(rr, 10)
    res["models"]["Reranker"] = summarize(rr, reordered[:, :10], split, n_boot)
    imp = ranker.booster_.feature_importance(importance_type="gain")
    res["reranker"] = {
        **rinfo, "n_candidates": n_cand,
        "candidate_recall_val": float(yv.reshape(-1, n_cand).any(axis=1).mean()),
        "candidate_recall_test": float((cands["test"] == split.test[:, None]).any(axis=1).mean()),
        "feature_gain_share": {f: float(g / imp.sum()) for f, g in
                               zip(FEATURES, imp, strict=True)},
    }

    res["paired_ndcg10"] = {
        f"{a} - {b}": paired_bootstrap(per_user[a], per_user[b], n_boot=n_boot)
        for a, b in (("SASRec", "EASE"), ("Reranker", "SASRec"), ("Reranker", "EASE"),
                     ("EASE", "ItemKNN"))
    }
    res["runtime_seconds"] = time.time() - t_start
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    return res
