"""Optuna search for the LightGBM LambdaRank re-ranker, chosen on validation, tested once.

The re-ranker is trained on the *validation* targets, so the validation stage is its training
data and cannot also score its hyperparameters. The search therefore splits the validation-stage
users (one fixed random split, seed 0) into a **fit** part (80%) and a **tune** part (20%):

* each trial runs the repo's own training routine (``train_ranker``: users whose validation film
  is among the candidates, 10% of them held out for early stopping) on the fit users only;
* the trial is scored by NDCG@10 of the re-ordered candidates on the tune users (users whose
  validation film is missing from the candidates count as a miss, as in the headline metric).

Every candidate, feature and label in the search comes from training histories and validation
targets. Test targets are never read, and test-stage features are only built after the search.
The fixed recipe (``DEFAULTS``) is enqueued as trial 0, so the selected recipe is at least as good
as the default *on the tune users*.

Both recipes are then run through the standard protocol once per seed (SASRec seed ``s`` supplies
the candidates and features, LightGBM seed ``s``): trained on all validation-stage users and
scored once on the test targets, with a paired bootstrap over users of tuned minus default.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import Split
from .metrics import hit, ndcg, paired_bootstrap
from .models import EASE, ItemKNN
from .pipeline import EASE_GRID, KNN_GRID, sasrec_for_seed, stage_scores, val_ndcg
from .rerank import (
    RANKER_DEFAULTS,
    StageInputs,
    build_features,
    merge_candidates,
    rerank_ranks,
    train_ranker,
)
from .sasrec import SASRecConfig

DEFAULTS = dict(RANKER_DEFAULTS)
# Ranges searched; every range contains the default. Scale-type parameters are searched on a log
# scale; ``lambda_l2`` (default 0, which a log scale cannot hold) on a log1p scale.
SPACE = {
    "num_leaves": (7, 255),
    "learning_rate": (0.01, 0.3),
    "min_child_samples": (5, 500),
    "feature_fraction": (0.4, 1.0),
    "bagging_fraction": (0.4, 1.0),
    "bagging_freq": (1, 10),
    "lambda_l2": (0.0, 100.0),
    "n_estimators": (200, 2000),
}
LOG = {"num_leaves", "learning_rate", "min_child_samples", "n_estimators"}
INTS = {"num_leaves", "min_child_samples", "bagging_freq", "n_estimators"}


def suggest_params(trial) -> dict:
    """Sample one recipe; ``lambda_l2`` is drawn as ``log1p_lambda_l2`` so that 0 is reachable."""
    out = {}
    for k, (lo, hi) in SPACE.items():
        if k == "lambda_l2":
            out[k] = float(np.expm1(trial.suggest_float("log1p_lambda_l2", np.log1p(lo),
                                                        np.log1p(hi))))
        elif k in INTS:
            out[k] = trial.suggest_int(k, lo, hi, log=k in LOG)
        else:
            out[k] = trial.suggest_float(k, lo, hi, log=k in LOG)
    return out


def to_trial_params(params: dict) -> dict:
    """Inverse of ``suggest_params`` naming, used to enqueue the default recipe."""
    out = {k: v for k, v in params.items() if k != "lambda_l2"}
    out["log1p_lambda_l2"] = float(np.log1p(params["lambda_l2"]))
    return out


def from_trial_params(tp: dict) -> dict:
    out = {k: v for k, v in tp.items() if k != "log1p_lambda_l2"}
    out["lambda_l2"] = float(np.expm1(tp["log1p_lambda_l2"]))
    return {**DEFAULTS, **out}


def split_users(n_users: int, tune_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Fixed random user split of the validation stage into (fit users, tune users)."""
    perm = np.random.default_rng(seed).permutation(n_users)
    n_tune = max(1, int(round(tune_frac * n_users)))
    return np.sort(perm[n_tune:]), np.sort(perm[:n_tune])


def rows_of(users: np.ndarray, n_cand: int) -> np.ndarray:
    return (users[:, None] * n_cand + np.arange(n_cand)[None, :]).ravel()


def label_ranks(pred: np.ndarray, y: np.ndarray, n_cand: int, miss_rank: int) -> np.ndarray:
    """0-based rank of the (single) positive after sorting each group by ``pred``."""
    p = pred.reshape(-1, n_cand)
    lab = y.reshape(-1, n_cand)
    order = np.argsort(-p, axis=1, kind="stable")
    pos = np.take_along_axis(lab, order, axis=1) > 0
    return np.where(pos.any(axis=1), pos.argmax(axis=1), miss_rank)


def tune_score(params: dict, xv: np.ndarray, yv: np.ndarray, n_cand: int, fit: np.ndarray,
               tune: np.ndarray, miss_rank: int, seed: int = 0,
               n_jobs: int | None = None) -> tuple[float, int]:
    """Train on the fit users (repo routine), return (tune-user NDCG@10, trees used)."""
    rf, rt = rows_of(fit, n_cand), rows_of(tune, n_cand)
    model, info = train_ranker(xv[rf], yv[rf], n_cand, seed=seed, params=params, n_jobs=n_jobs)
    ranks = label_ranks(model.predict(xv[rt]), yv[rt], n_cand, miss_rank)
    return float(ndcg(ranks, 10).mean()), info["best_iteration"]


def tune_ranker(xv: np.ndarray, yv: np.ndarray, n_cand: int, fit: np.ndarray, tune: np.ndarray,
                miss_rank: int, n_trials: int = 100, sampler_seed: int = 0,
                model_seed: int = 0, n_jobs: int | None = None, timeout: float | None = None,
                verbose: bool = True) -> dict:
    """TPE search maximising tune-user NDCG@10. Reads validation-stage arrays only."""
    import optuna

    if np.intersect1d(fit, tune).size:
        raise ValueError("fit and tune users overlap")

    def objective(trial) -> float:
        params = suggest_params(trial)
        t0 = time.perf_counter()
        value, trees = tune_score(params, xv, yv, n_cand, fit, tune, miss_rank,
                                  seed=model_seed, n_jobs=n_jobs)
        trial.set_user_attr("trees", trees)
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=sampler_seed))
    study.enqueue_trial(to_trial_params(DEFAULTS))

    def report(study, trial) -> None:
        if verbose and (trial.number % 10 == 0 or trial.number == n_trials - 1):
            print(f"  trial {trial.number:3d} tune NDCG@10 {trial.value:.4f} "
                  f"(best {study.best_value:.4f}, {trial.user_attrs['trees']} trees, "
                  f"{trial.user_attrs['seconds']:.1f}s)", flush=True)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[report])
    seconds = time.perf_counter() - t0
    trials = [{"number": t.number, "tune_ndcg10": t.value,
               "params": from_trial_params(t.params),
               "trees": t.user_attrs.get("trees"), "seconds": t.user_attrs.get("seconds")}
              for t in study.trials if t.value is not None]
    best = study.best_trial
    return {
        "best_params": from_trial_params(best.params),
        "best_trial": best.number,
        "best_tune_ndcg10": float(best.value),
        "best_trees": best.user_attrs["trees"],
        "default_tune_ndcg10": float(trials[0]["tune_ndcg10"]),
        "default_trees": trials[0]["trees"],
        "n_trials": len(trials),
        "seconds": seconds,
        "trials": trials,
    }


def _stage_features(split: Split, stage: str, ease: np.ndarray, sas: np.ndarray,
                    knn: np.ndarray, n_cand: int) -> tuple[np.ndarray, np.ndarray]:
    inp = StageInputs(ease=ease, sas=sas, knn=knn)
    cands = merge_candidates([ease, sas], n_cand)
    return build_features(split, stage, inp, cands), cands


def run_tuning(split: Split, out_dir: str | Path, sasrec_cfg: SASRecConfig | None = None,
               seeds: tuple[int, ...] = (0, 1, 2), cache_dir: str | Path | None = None,
               n_trials: int = 100, n_cand: int = 100, n_boot: int = 1000,
               n_jobs: int | None = None, timeout: float | None = None,
               tune_frac: float = 0.2, ease_grid=EASE_GRID, knn_grid=KNN_GRID,
               verbose: bool = True) -> dict:
    """Search on validation (SASRec seed ``seeds[0]``), then score default vs tuned on test."""
    t_all = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)
    base_cfg = sasrec_cfg or SASRecConfig()
    cache = Path(cache_dir) if cache_dir else None
    miss = split.n_items

    # retrievers: same validation-only selection as run_pipeline
    knn_res = {k: val_ndcg(ItemKNN(k=k).fit(split.train, split.n_items), split) for k in knn_grid}
    best_k = max(knn_res, key=knn_res.get)
    ease_res = {lam: val_ndcg(EASE(lam).fit(split.train, split.n_items), split)
                for lam in ease_grid}
    best_lam = max(ease_res, key=ease_res.get)
    ease_m = EASE(best_lam).fit(split.train, split.n_items)
    knn_m = ItemKNN(k=best_k).fit(split.train, split.n_items)
    sas = {s: sasrec_for_seed(split, SASRecConfig(**{**asdict(base_cfg), "seed": s}), cache)
           for s in seeds}
    log(f"retrievers ready: EASE lambda={best_lam}, ItemKNN k={best_k}, SASRec seeds {seeds}")

    # ---- search: validation stage only -------------------------------------------------
    s0 = seeds[0]
    ease_v, knn_v = stage_scores(ease_m, split, "val"), stage_scores(knn_m, split, "val")
    xv0, cv0 = _stage_features(split, "val", ease_v, stage_scores(sas[s0], split, "val"),
                               knn_v, n_cand)
    yv0 = (cv0 == split.val[:, None]).ravel().astype(int)
    fit, tune = split_users(split.n_users, tune_frac, seed=0)
    log(f"search: {len(fit)} fit users, {len(tune)} tune users, {n_trials} trials")
    search = tune_ranker(xv0, yv0, n_cand, fit, tune, miss, n_trials=n_trials,
                         model_seed=s0, n_jobs=n_jobs, timeout=timeout, verbose=verbose)
    recipes = {"default": dict(DEFAULTS), "tuned": search["best_params"]}

    # selection check: both recipes on the tune users with every LightGBM seed
    val_check = {arm: [tune_score(p, xv0, yv0, n_cand, fit, tune, miss, seed=s,
                                  n_jobs=n_jobs)[0] for s in seeds]
                 for arm, p in recipes.items()}

    # ---- final: train on all validation-stage users, score test once per seed ----------
    ease_t, knn_t = stage_scores(ease_m, split, "test"), stage_scores(knn_m, split, "test")
    per_seed, per_user = [], {arm: [] for arm in recipes}
    for s in seeds:
        xv, cv = _stage_features(split, "val", ease_v, stage_scores(sas[s], split, "val"),
                                 knn_v, n_cand)
        yv = (cv == split.val[:, None]).ravel().astype(int)
        xt, ct = _stage_features(split, "test", ease_t, stage_scores(sas[s], split, "test"),
                                 knn_t, n_cand)
        row = {"seed": s}
        nd = {}
        for arm, p in recipes.items():
            model, info = train_ranker(xv, yv, n_cand, seed=s, params=p, n_jobs=n_jobs)
            r, _ = rerank_ranks(model.predict(xt), ct, split.test, miss)
            nd[arm] = ndcg(r, 10)
            per_user[arm].append(nd[arm])
            row[arm] = {"ndcg10": float(nd[arm].mean()), "hr10": float(hit(r, 10).mean()),
                        "trees": info["best_iteration"]}
        row["tuned_minus_default_ndcg10"] = paired_bootstrap(nd["tuned"], nd["default"],
                                                             n_boot=n_boot)
        per_seed.append(row)
        log(f"seed {s}: default {row['default']['ndcg10']:.4f}, tuned {row['tuned']['ndcg10']:.4f}")

    mean_user = {arm: np.mean(v, axis=0) for arm, v in per_user.items()}
    pooled = {
        arm: {"ndcg10_mean": float(np.mean([r[arm]["ndcg10"] for r in per_seed])),
              "ndcg10_sd": float(np.std([r[arm]["ndcg10"] for r in per_seed], ddof=1))
              if len(per_seed) > 1 else 0.0,
              "hr10_mean": float(np.mean([r[arm]["hr10"] for r in per_seed]))}
        for arm in recipes
    }
    result = {
        "protocol": {
            "search_data": "validation stage only: candidates/features from training histories, "
                           "labels = validation film; test targets never read during the search",
            "search_split": f"validation-stage users split at random (seed 0): "
                            f"{len(fit)} fit users, {len(tune)} tune users",
            "objective": "NDCG@10 of the re-ordered candidates on the tune users (missing "
                         "validation film = miss), ranker trained on the fit users with the "
                         "repo routine (10% of fit users with a positive for early stopping, "
                         "patience 50)",
            "sampler": f"TPE (seed 0); trial 0 = default recipe; no pruning; search features "
                       f"from SASRec seed {s0}, LightGBM seed {s0}",
            "final": "ranker trained on all validation-stage users (repo routine), scored once "
                     "on the test targets; seed s = SASRec seed s features + LightGBM seed s",
            "seeds": list(seeds),
            "bootstrap": f"{n_boot} paired resamples of users, tuned minus default",
            "n_jobs": n_jobs,
            "n_candidates": n_cand,
            "retrievers": {"ease_lambda": float(best_lam), "itemknn_k": int(best_k)},
            "total_seconds": time.perf_counter() - t_all,
        },
        "space": {k: list(v) for k, v in SPACE.items()},
        "default_params": dict(DEFAULTS),
        "tuned_params": search["best_params"],
        "search": search,
        "validation_check": {
            "note": "tune-user NDCG@10, ranker trained on fit users, SASRec seed "
                    f"{s0} features, LightGBM seeds {list(seeds)}",
            **{arm: {"per_seed": v, "mean": float(np.mean(v))} for arm, v in val_check.items()},
        },
        "test": {
            "per_seed": per_seed,
            "pooled": pooled,
            "tuned_minus_default_ndcg10_seed_mean": paired_bootstrap(
                mean_user["tuned"], mean_user["default"], n_boot=n_boot),
        },
    }
    (out / "tuning.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "tuning.md").write_text(tuning_markdown(result), encoding="utf-8")
    return result


def _ci(d: dict, nd: int = 4) -> str:
    return f"{d['diff']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"


def _fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def tuning_markdown(result: dict) -> str:
    """Human-readable summary of ``tuning.json``."""
    p, s, t, vc = result["protocol"], result["search"], result["test"], result["validation_check"]
    lines = [
        "# Does tuning help? Optuna search for the LightGBM re-ranker",
        "",
        f"Search: {p['sampler']}. Data: {p['search_data']}. Split: {p['search_split']}. "
        f"Objective: {p['objective']}. Final: {p['final']}. Bootstrap: {p['bootstrap']}. "
        f"Threads: {p['n_jobs']}. Total time {p['total_seconds'] / 60:.1f} min.",
        "",
        f"{s['n_trials']} trials in {s['seconds'] / 60:.1f} min. Tune-user NDCG@10 (search seed): "
        f"default {s['default_tune_ndcg10']:.4f} ({s['default_trees']} trees), best "
        f"{s['best_tune_ndcg10']:.4f} (trial {s['best_trial']}, {s['best_trees']} trees). "
        f"Re-checked over LightGBM seeds {p['seeds']}: default {vc['default']['mean']:.4f}, "
        f"tuned {vc['tuned']['mean']:.4f}.",
        "",
        "| Parameter | Default | Tuned | Range searched |",
        "|---|---|---|---|",
    ]
    for k, rng in result["space"].items():
        lines.append(f"| {k} | {_fmt(result['default_params'][k])} "
                     f"| {_fmt(result['tuned_params'][k])} | {rng[0]}-{rng[1]} |")
    lines += [
        "",
        "Test (6,040 users when run on MovieLens-1M), per seed:",
        "",
        "| Seed | Default NDCG@10 | Tuned NDCG@10 | Default HR@10 | Tuned HR@10 "
        "| Trees (default / tuned) | Tuned - default NDCG@10 [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in t["per_seed"]:
        lines.append(
            f"| {r['seed']} | {r['default']['ndcg10']:.4f} | {r['tuned']['ndcg10']:.4f} "
            f"| {r['default']['hr10']:.4f} | {r['tuned']['hr10']:.4f} "
            f"| {r['default']['trees']} / {r['tuned']['trees']} "
            f"| {_ci(r['tuned_minus_default_ndcg10'])} |")
    pd_, pt = t["pooled"]["default"], t["pooled"]["tuned"]
    lines += [
        "",
        f"Mean over seeds: default NDCG@10 {pd_['ndcg10_mean']:.4f} ± {pd_['ndcg10_sd']:.4f}, "
        f"tuned {pt['ndcg10_mean']:.4f} ± {pt['ndcg10_sd']:.4f} (mean ± SD). Tuned minus "
        f"default on each user's seed-averaged NDCG@10: "
        f"{_ci(t['tuned_minus_default_ndcg10_seed_mean'])}.",
        "",
    ]
    return "\n".join(lines)
