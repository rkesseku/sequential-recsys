import numpy as np

from seqrec.metrics import (
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


def test_mask_seen_and_ranks():
    s = np.array([[0.0, 5.0, 4.0, 3.0, 2.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
    m = mask_seen(s, [[1], []])
    assert np.isneginf(m[0, 0]) and np.isneginf(m[0, 1]) and np.isneginf(m[1, 0])
    r = target_ranks(m, np.array([2, 1]))
    assert r.tolist() == [0, 3]


def test_ties_are_pessimistic():
    s = np.array([[-np.inf, 1.0, 1.0, 1.0]])
    assert target_ranks(s, np.array([1]))[0] == 2


def test_ndcg_and_hit_values():
    r = np.array([0, 1, 9, 10])
    assert hit(r, 10).tolist() == [1, 1, 1, 0]
    np.testing.assert_allclose(ndcg(r, 10), [1.0, 1 / np.log2(3), 1 / np.log2(11), 0.0])


def test_topk_order():
    s = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert topk(s, 3).tolist() == [[1, 3, 2]]


def test_bootstrap_contains_mean():
    v = np.random.default_rng(0).random(500)
    m, lo, hi = bootstrap_ci(v, n_boot=200)
    assert lo <= m <= hi
    d = paired_bootstrap(v + 0.1, v, n_boot=200)
    assert abs(d["diff"] - 0.1) < 1e-9 and d["p_le_zero"] == 0.0


def test_coverage_and_popularity():
    top = np.array([[1, 2], [1, 3]])
    assert coverage(top, 10) == 0.3
    counts = np.array([0, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    st = popularity_stats(top, counts)
    assert st["head_share"] == 0.5
    assert st["arp"] == (100 + 1 + 100 + 1) / 4
