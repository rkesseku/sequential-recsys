"""Figures for the README and brief, drawn only from metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["Popularity", "ItemKNN", "EASE", "SASRec", "Reranker"]
COLORS = {"Popularity": "#9aa5b1", "ItemKNN": "#7fa7d9", "EASE": "#3f73c4",
          "SASRec": "#1f4e9e", "Reranker": "#d9822b"}
OFFSETS = {"SASRec": (6, -22), "Reranker": (8, 2), "ItemKNN": (8, -14), "EASE": (8, 2)}


def _models(m: dict) -> list[str]:
    return [k for k in ORDER if k in m["models"]]


def accuracy_bars(m: dict, path: Path) -> None:
    names = _models(m)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, label in ((axes[0], "ndcg10", "NDCG@10"), (axes[1], "hr10", "HR@10")):
        vals = [m["models"][n][key] for n in names]
        mean = [v["mean"] for v in vals]
        err = [[v["mean"] - v["lo"] for v in vals], [v["hi"] - v["mean"] for v in vals]]
        ax.bar(names, mean, yerr=err, capsize=3, color=[COLORS[n] for n in names])
        ax.set_title(f"{label} (test, 95% bootstrap CI)", fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def recall_curve(m: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for n in _models(m):
        rc = m["models"][n]["recall_curve"]
        ks = [int(k) for k in rc]
        ax.plot(ks, list(rc.values()), marker="o", ms=3, label=n, color=COLORS[n])
    ax.set_xscale("log")
    ax.set_xlabel("K (log scale)")
    ax.set_ylabel("Recall@K (= HR@K)")
    ax.set_title("Share of users whose next film is in the top K", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sasrec_training(m: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for seed, curve in m["sasrec_curves"].items():
        pts = [(c["epoch"], c["val_ndcg10"]) for c in curve if "val_ndcg10" in c]
        if pts:
            ax.plot(*zip(*pts, strict=True), marker=".", label=f"seed {seed}")
    ease = m["tuning"]["ease_lambda_val_ndcg10"][str(m["tuning"]["ease_best_lambda"])]
    ax.axhline(ease, ls="--", color=COLORS["EASE"], label="EASE (val)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NDCG@10")
    ax.set_title("SASRec early stopping on validation NDCG@10", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def accuracy_vs_popularity(m: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for n in _models(m):
        d = m["models"][n]
        ax.scatter(d["head_share10"], d["ndcg10"]["mean"], s=60, color=COLORS[n])
        ax.annotate(f"{n}\ncov {d['coverage10']:.0%}", (d["head_share10"], d["ndcg10"]["mean"]),
                    textcoords="offset points", xytext=OFFSETS.get(n, (6, -4)), fontsize=7)
    ax.margins(x=0.12, y=0.1)
    ax.set_xlabel("share of top-10 slots given to the 10% most popular films")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Accuracy vs. popularity concentration", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_all(metrics_path: str | Path, out_dir: str | Path) -> list[Path]:
    m = json.loads(Path(metrics_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn, name in ((accuracy_bars, "accuracy.png"), (recall_curve, "recall_curve.png"),
                     (sasrec_training, "sasrec_training.png"),
                     (accuracy_vs_popularity, "accuracy_vs_popularity.png")):
        p = out / name
        fn(m, p)
        paths.append(p)
    return paths
