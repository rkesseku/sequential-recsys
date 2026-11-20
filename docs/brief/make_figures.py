"""Build the brief's figures: an architecture diagram plus copies of the result plots."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE / "figures"
RESULTS = HERE.parent / "results"


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=color, ec="#1f4e9e", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "->", "color": "#333", "lw": 1.1})


def architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.15, 4)
    ax.axis("off")
    box(ax, 0.1, 1.5, 1.7, 1.0, "MovieLens-1M\n1.0M ratings\n6,040 users", "#eef3fb")
    box(ax, 2.2, 1.5, 1.8, 1.0, "Chronological\nleave-last-out\ntrain | val | test", "#eef3fb")
    # Every retriever is scored on test (dashed frame -> evaluation), and ItemKNN scores and
    # ranks are re-ranker features (rerank.FEATURES), so the baseline box also feeds the merge.
    ax.add_patch(FancyBboxPatch((4.37, 1.0), 2.26, 2.92, boxstyle="round,pad=0.0,rounding_size=0.05",
                                fc="none", ec="#7f8fa6", lw=1.0, ls="--"))
    box(ax, 4.45, 3.1, 2.1, 0.7, "Popularity, ItemKNN\n(baselines)", "#f4f6f8")
    box(ax, 4.45, 2.1, 2.1, 0.7, "EASE\n(closed-form, tuned on val)", "#dbe6f7")
    box(ax, 4.45, 1.1, 2.1, 0.7, "SASRec transformer\n(3 seeds, early stop)", "#dbe6f7")
    box(ax, 7.0, 1.4, 1.35, 1.2, "Merge top-100\ncandidates\n+ 10 features", "#fdf0e3")
    box(ax, 8.6, 1.4, 1.3, 1.2, "LightGBM\nLambdaRank\n(trained on val)", "#fdf0e3")
    box(ax, 2.2, -0.1, 7.7, 0.7,
        "Full-catalogue evaluation on the held-out last item: HR@10, NDCG@10, Recall@K,\n"
        "catalogue coverage, popularity share, bootstrap CIs over users", "#eef7ee")
    arrow(ax, 1.8, 2.0, 2.2, 2.0)
    for y in (3.45, 2.45, 1.45):
        arrow(ax, 4.0, 2.0, 4.45, y)
    arrow(ax, 6.55, 3.45, 7.0, 2.5)
    arrow(ax, 6.55, 2.45, 7.0, 2.2)
    arrow(ax, 6.55, 1.45, 7.0, 1.8)
    arrow(ax, 5.5, 1.0, 5.5, 0.62)
    arrow(ax, 8.35, 2.0, 8.6, 2.0)
    arrow(ax, 9.25, 1.4, 9.25, 0.62)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    architecture(FIG / "architecture.png")
    for name in ("accuracy.png", "recall_curve.png", "sasrec_training.png",
                 "accuracy_vs_popularity.png"):
        src = RESULTS / name
        if src.exists():
            shutil.copy(src, FIG / name)


if __name__ == "__main__":
    main()
