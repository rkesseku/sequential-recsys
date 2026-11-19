import numpy as np
import pandas as pd
import pytest

from seqrec.data import build_split, interaction_matrix


def test_chronological_leave_last_out_with_stable_ties():
    df = pd.DataFrame({
        "user": [1, 1, 1, 1, 2, 2, 2],
        "item": [30, 10, 20, 40, 10, 20, 30],
        "rating": 5,
        "ts": [3, 1, 1, 4, 5, 6, 7],
    })
    s = build_split(df)
    # raw ids 10,20,30,40 -> 1,2,3,4; user 1 order by ts with ties in file order: 10,20,30,40
    assert s.train[0] == [1, 2]
    assert (s.val[0], s.test[0]) == (3, 4)
    assert s.train[1] == [1] and (s.val[1], s.test[1]) == (2, 3)
    assert s.history("test", 0) == [1, 2, 3]
    assert s.n_items == 4


def test_short_users_dropped_and_ids_contiguous(split):
    df = pd.DataFrame({"user": [1, 1, 2, 2, 2], "item": [5, 6, 5, 7, 8], "rating": 1,
                       "ts": [0, 1, 0, 1, 2]})
    s = build_split(df, min_len=3)
    assert s.n_users == 1 and s.n_items == 3
    all_items = {i for h in split.train for i in h} | set(split.val) | set(split.test)
    assert min(all_items) >= 1 and max(all_items) <= split.n_items


def test_bad_stage_raises(split):
    with pytest.raises(ValueError):
        split.history("train", 0)
    with pytest.raises(ValueError):
        split.targets("oops")


def test_interaction_matrix_and_counts(split):
    x = interaction_matrix([[1, 2, 2], [3]], 4)
    assert x.shape == (2, 5)
    assert x.toarray().tolist() == [[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]
    c = split.item_counts("val")
    assert c[0] == 0 and c.sum() == sum(len(h) for h in split.train)
    assert np.all(split.item_counts("test") >= c)
