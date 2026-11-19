import numpy as np

from seqrec.metrics import ndcg, target_ranks
from seqrec.models import EASE, ItemKNN, Popularity
from seqrec.pipeline import stage_scores, val_ndcg
from seqrec.sasrec import SASRec


def _test_ndcg(model, split):
    return float(ndcg(target_ranks(stage_scores(model, split, "test"), split.test), 10).mean())


def test_shapes_and_seen_items_excluded(split):
    for m in (Popularity(), ItemKNN(k=10), EASE(10.0)):
        m.fit(split.train, split.n_items)
        s = stage_scores(m, split, "val")
        assert s.shape == (split.n_users, split.n_items + 1)
        assert np.isneginf(s[0, split.train[0]]).all() and np.isneginf(s[:, 0]).all()


def test_ease_weights_have_zero_diagonal(split):
    m = EASE(10.0).fit(split.train, split.n_items)
    assert np.allclose(np.diag(m.b_), 0.0)


def test_collaborative_models_beat_popularity(split):
    pop = _test_ndcg(Popularity().fit(split.train, split.n_items), split)
    ease = _test_ndcg(EASE(10.0).fit(split.train, split.n_items), split)
    knn = _test_ndcg(ItemKNN(k=10).fit(split.train, split.n_items), split)
    assert ease > pop and knn > pop


def test_sasrec_learns_sequential_pattern(split, tiny_cfg):
    m = SASRec(tiny_cfg).fit(split.train, split.n_items, val_fn=lambda x: val_ndcg(x, split))
    assert m.best_val_ > 0
    assert any("val_ndcg10" in h for h in m.history_)
    s = m.score(split.histories("test"))
    assert s.shape == (split.n_users, split.n_items + 1) and np.isfinite(s).all()
    pop = _test_ndcg(Popularity().fit(split.train, split.n_items), split)
    assert _test_ndcg(m, split) > pop
