"""Command-line entry point: ``seqrec run-all``, ``seqrec train-sasrec``, ``seqrec plots``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cfg(args: argparse.Namespace, seed: int):
    from .sasrec import SASRecConfig  # lazy: `seqrec plots` should not need torch

    return SASRecConfig(max_len=args.max_len, dim=args.dim, epochs=args.epochs,
                        patience=args.patience, seed=seed, threads=args.threads)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="seqrec")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run-all", "train-sasrec"):
        s = sub.add_parser(name)
        s.add_argument("--data", default="data/raw/ml-1m")
        s.add_argument("--cache", default="results/cache")
        s.add_argument("--epochs", type=int, default=40)
        s.add_argument("--patience", type=int, default=3)
        s.add_argument("--max-len", type=int, default=100)
        s.add_argument("--dim", type=int, default=64)
        s.add_argument("--threads", type=int, default=None)
        s.add_argument("--verbose", action="store_true")
        if name == "run-all":
            s.add_argument("--out", default="docs/results")
            s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
            s.add_argument("--n-boot", type=int, default=1000)
        else:
            s.add_argument("--seed", type=int, default=0)
    pl = sub.add_parser("plots")
    pl.add_argument("--metrics", default="docs/results/metrics.json")
    pl.add_argument("--out", default="docs/results")
    args = p.parse_args(argv)

    if args.cmd == "plots":
        from .plots import make_all

        for path in make_all(args.metrics, args.out):
            print(path)
        return

    from .data import load_movielens

    split = load_movielens(args.data)
    if args.cmd == "train-sasrec":
        from .pipeline import sasrec_for_seed

        m = sasrec_for_seed(split, _cfg(args, args.seed), Path(args.cache), verbose=args.verbose)
        print(json.dumps({"seed": args.seed, "best_val_ndcg10": m.best_val_,
                          "epochs": len(m.history_), "seconds": m.train_seconds_}))
        return

    from .pipeline import run_pipeline
    from .plots import make_all

    res = run_pipeline(split, args.out, sasrec_cfg=_cfg(args, args.seeds[0]),
                       seeds=tuple(args.seeds), cache_dir=args.cache, n_boot=args.n_boot,
                       verbose=args.verbose)
    make_all(Path(args.out) / "metrics.json", args.out)
    for name, d in res["models"].items():
        print(f"{name:10s} NDCG@10={d['ndcg10']['mean']:.4f} HR@10={d['hr10']['mean']:.4f} "
              f"R@50={d['recall50']:.4f} cov={d['coverage10']:.3f}")


if __name__ == "__main__":
    main()
