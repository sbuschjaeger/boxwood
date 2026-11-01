"""Backend selection: every backend matches the scalar backend bit-exactly."""

import numpy as np
import pytest
from conftest import train_catboost

import boxwood as bw
from test_catboost_ref import CATS, make_data


def _train(X, y):
    """A tiny binary model with categorical features and CTR combinations."""
    return train_catboost(X, y, CATS, loss_function="Logloss", max_ctr_complexity=2)


def test_backends_match_scalar():
    X, y = make_data(n=800)
    ote = bw.ObliviousTreeEnsemble.from_catboost(_train(X, y))
    assert bw.backends()  # scalar is always available
    ote.set_backend("scalar")
    base = ote.predict_margin(X)
    for name in bw.backends():
        ote.set_backend(name)
        assert np.array_equal(ote.predict_margin(X), base)  # bit-exact
    ote.set_backend("")  # best backend
    assert np.array_equal(ote.predict_margin(X), base)


def test_unknown_backend():
    X, y = make_data(n=100)
    ote = bw.ObliviousTreeEnsemble.from_catboost(_train(X, y))
    with pytest.raises(ValueError):
        ote.set_backend("nope")


def test_threads_match_one_thread():
    X, y = make_data(n=3000)
    ote = bw.ObliviousTreeEnsemble.from_catboost(_train(X, y))
    base = ote.predict_margin(X)
    leaves = ote.leaf_index(X)
    for n in (2, 8):
        ote.set_threads(n)
        assert np.array_equal(ote.predict_margin(X), base)
        assert np.array_equal(ote.leaf_index(X), leaves)
    with pytest.raises(ValueError):
        ote.set_threads(-1)
