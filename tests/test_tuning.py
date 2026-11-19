"""Optuna re-ranker study on tiny synthetic logs (no data files, no network)."""

import json
from dataclasses import replace

import numpy as np
import pytest

from seqrec.cli import main
from seqrec.data import synthetic_log
from seqrec.tuning import DEFAULTS, SPACE, run_tuning

pytest.importorskip("optuna")

KW = dict(n_trials=4, n_cand=10, n_boot=50, n_jobs=1, ease_grid=(10.0,), knn_grid=(10,),
          verbose=False)


@pytest.fixture(scope="module")
def tuned(split, tmp_path_factory):
    from seqrec.sasrec import SASRecConfig

    cfg = SASRecConfig(max_len=16, dim=16, n_blocks=1, dropout=0.1, lr=5e-3, batch_size=32,
                       epochs=6, patience=2, eval_every=2, seed=0, threads=1)
    root = tmp_path_factory.mktemp("tune")
    res = run_tuning(split, root / "out", sasrec_cfg=cfg, seeds=(0, 1), cache_dir=root / "cache",
                     **KW)
    return res, cfg, root


def test_trial0_is_default_and_tuned_not_worse_on_validation(tuned):
    res, _, root = tuned
    s = res["search"]
    assert s["trials"][0]["params"] == DEFAULTS
    assert s["default_tune_ndcg10"] == s["trials"][0]["tune_ndcg10"]
    assert s["best_tune_ndcg10"] >= s["default_tune_ndcg10"]
    for k, v in res["tuned_params"].items():
        lo, hi = SPACE[k]
        assert lo <= v <= hi
    saved = json.loads((root / "out" / "tuning.json").read_text())
    assert len(saved["test"]["per_seed"]) == 2
    assert "Does tuning help?" in (root / "out" / "tuning.md").read_text()


def test_corrupting_test_targets_does_not_change_the_search(split, tuned):
    res, cfg, root = tuned
    rng = np.random.default_rng(5)
    bad = replace(split, test=rng.integers(1, split.n_items + 1, size=split.n_users))
    res2 = run_tuning(bad, root / "out2", sasrec_cfg=cfg, seeds=(0, 1), cache_dir=root / "cache",
                      **KW)

    def strip(r):
        return [{k: v for k, v in t.items() if k != "seconds"} for t in r["search"]["trials"]]

    assert strip(res2) == strip(res)
    assert res2["tuned_params"] == res["tuned_params"]
    assert res2["test"]["per_seed"] != res["test"]["per_seed"]  # the corruption did bite


def test_cli_tune_end_to_end(tmp_path, capsys):
    data = tmp_path / "ml"
    data.mkdir()
    log = synthetic_log(n_users=60, n_items=30, seq_len=12, seed=3)
    (data / "ratings.dat").write_text(
        "\n".join("::".join(map(str, r)) for r in log.itertuples(index=False)) + "\n")
    (data / "movies.dat").write_text(
        "\n".join(f"{i}::Film {i}::{'A' if i % 2 else 'B|C'}" for i in range(1, 31)) + "\n")
    main(["tune", "--data", str(data), "--cache", str(tmp_path / "cache"),
          "--out", str(tmp_path / "out"), "--epochs", "4", "--patience", "1", "--max-len", "16",
          "--dim", "16", "--threads", "1", "--seeds", "0", "--n-boot", "20", "--trials", "3",
          "--n-jobs", "1", "--n-cand", "10"])
    assert "test tuned-default" in capsys.readouterr().out
    saved = json.loads((tmp_path / "out" / "tuning.json").read_text())
    assert saved["search"]["n_trials"] == 3
