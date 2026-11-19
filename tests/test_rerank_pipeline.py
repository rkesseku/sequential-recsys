import json

import numpy as np

from seqrec.cli import main
from seqrec.pipeline import run_pipeline
from seqrec.plots import make_all
from seqrec.rerank import full_ranks, merge_candidates, rerank_ranks


def test_full_ranks_is_inverse_argsort():
    s = np.array([[0.2, 0.9, 0.1, 0.5]])
    assert full_ranks(s).tolist() == [[2, 0, 3, 1]]


def test_merge_candidates_interleaves_and_dedupes():
    a = np.array([[0.0, 9, 8, 7, 1, 0]])
    b = np.array([[0.0, 9, 1, 2, 3, 8]])
    c = merge_candidates([a, b], 4)
    assert c.tolist() == [[1, 2, 5, 3]]


def test_rerank_ranks_handles_misses():
    cands = np.array([[5, 6, 7], [1, 2, 3]])
    pred = np.array([0.1, 0.9, 0.5, 0.3, 0.2, 0.1])
    ranks, reordered = rerank_ranks(pred, cands, np.array([7, 9]), miss_rank=99)
    assert reordered[0].tolist() == [6, 7, 5]
    assert ranks.tolist() == [1, 99]


def test_run_pipeline_end_to_end(split, tiny_cfg, tmp_path):
    res = run_pipeline(split, tmp_path, sasrec_cfg=tiny_cfg, seeds=(0, 1),
                       cache_dir=tmp_path / "cache", n_boot=50, n_cand=20,
                       ease_grid=(5.0, 20.0), knn_grid=(5, 10))
    assert set(res["models"]) == {"Popularity", "ItemKNN", "EASE", "SASRec", "Reranker"}
    for d in res["models"].values():
        assert d["ndcg10"]["lo"] <= d["ndcg10"]["mean"] <= d["ndcg10"]["hi"]
        assert 0 <= d["coverage10"] <= 1
    assert len(res["sasrec_seeds"]) == 2
    assert (tmp_path / "cache" / "sasrec_seed0.pt").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["reranker"]["n_candidates"] == 20
    assert abs(sum(saved["reranker"]["feature_gain_share"].values()) - 1) < 1e-6
    paths = make_all(tmp_path / "metrics.json", tmp_path / "fig")
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    # second run reuses cached checkpoints and gives identical SASRec numbers
    res2 = run_pipeline(split, tmp_path / "again", sasrec_cfg=tiny_cfg, seeds=(0, 1),
                        cache_dir=tmp_path / "cache", n_boot=50, n_cand=20,
                        ease_grid=(5.0, 20.0), knn_grid=(5, 10))
    assert res2["sasrec_seeds"] == res["sasrec_seeds"]


def test_cli_plots(split, tiny_cfg, tmp_path, capsys):
    run_pipeline(split, tmp_path, sasrec_cfg=tiny_cfg, seeds=(0,), n_boot=20, n_cand=10,
                 ease_grid=(10.0,), knn_grid=(10,))
    main(["plots", "--metrics", str(tmp_path / "metrics.json"), "--out", str(tmp_path / "p")])
    assert "accuracy.png" in capsys.readouterr().out
