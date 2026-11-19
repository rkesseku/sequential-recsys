import os

import pytest

# Local Windows conda envs can load two OpenMP runtimes (MKL numpy + torch); harmless on CI.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from seqrec.data import build_split, synthetic_log  # noqa: E402
from seqrec.sasrec import SASRecConfig  # noqa: E402


@pytest.fixture(scope="session")
def split():
    genres = {i: ["A"] if i % 2 else ["B", "C"] for i in range(1, 41)}
    return build_split(synthetic_log(n_users=80, n_items=40, seq_len=14, seed=1),
                       raw_genres=genres)


@pytest.fixture
def tiny_cfg():
    return SASRecConfig(max_len=16, dim=16, n_blocks=1, dropout=0.1, lr=5e-3,
                        batch_size=32, epochs=30, patience=4, eval_every=2, seed=0)
