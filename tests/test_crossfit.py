"""The training table that decides what is tested must be scored out of sample."""
import sys

import numpy as np
import pytest

from caac.pool.protocol import build_table, concat_tables
from caac.pool.synth import make_pools


def _certify():
    sys.path.insert(0, ".")
    import certify
    return certify


def test_fold_estimators_never_see_their_held_out_questions():
    C = _certify()
    small, big = make_pools(60, n_small=8, n_big=2, seed=3)
    folds = C.fit_fold_estimators(small, big, (2, 4), folds=3)
    held = [q for h, _ in folds for q in h]
    assert sorted(held) == sorted(small) and len(set(held)) == len(held)


def test_crossfit_table_covers_every_training_question_once():
    C = _certify()
    small, big = make_pools(60, n_small=8, n_big=2, seed=3)
    ests = C.fit_estimators(small, big, (2, 4))
    folds = C.fit_fold_estimators(small, big, (2, 4), folds=3)
    bm = next(iter(big.values())).model
    fam = C.caac_family("caac_staged", ests, 1.5, bm, (2, 4))
    t = C.train_table("caac_staged", fam, (small, big), folds, 1.5, bm, (2, 4))
    assert sorted(t.qids) == sorted(small)
    assert t.served.shape == (60, len(fam.configs))


def test_baselines_use_the_plain_training_table():
    C = _certify()
    small, big = make_pools(30, n_small=8, n_big=2, seed=3)
    fam = C.baseline_family("sc_abstain", 2, 8)
    t = C.train_table("sc_abstain", fam, (small, big), [], 1.5, None, (2, 4))
    ref = build_table(fam, sorted(small), small, big)
    assert np.array_equal(t.served, ref.served) and t.qids == ref.qids


def test_concat_refuses_overlapping_or_mismatched_tables():
    C = _certify()
    small, big = make_pools(20, n_small=8, n_big=2, seed=3)
    fam = C.baseline_family("sc_abstain", 2, 8)
    q = sorted(small)
    a, b = build_table(fam, q[:10], small, big), build_table(fam, q[10:], small, big)
    whole = concat_tables([a, b])
    assert whole.qids == q and whole.served.shape[0] == 20
    with pytest.raises(ValueError):
        concat_tables([a, a])
