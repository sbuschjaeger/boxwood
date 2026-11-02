"""Backend selection: every backend matches the scalar backend bit-exactly."""

import numpy as np
import pandas as pd
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


def _wide(n_features):
    """A numeric regression model whose splits use about `n_features` features."""
    rng = np.random.default_rng(n_features)
    X = pd.DataFrame(rng.normal(size=(600, n_features)).astype(np.float32))
    return X, train_catboost(X, X.sum(axis=1), loss_function="RMSE", iterations=150, depth=6)


@pytest.mark.parametrize("model", ["categorical", "wide_200", "wide_300"])
def test_small_batches_match_full_batch(model):
    # Batches below a vector group take each backend's single-observation kernel; models with more than
    # 128 and more than 256 split features take its other paths.
    if model == "categorical":
        X, y = make_data(n=300)
        cb = _train(X, y)
    else:
        X, cb = _wide(int(model.split("_")[1]))
    ote = bw.ObliviousTreeEnsemble.from_catboost(cb)
    for leaf_type in ("float32", "int8"):
        ote.set_leaf_type(leaf_type)
        ote.set_backend("scalar")
        base, leaves = ote.predict_margin(X), ote.leaf_index(X)
        for name in bw.backends():
            ote.set_backend(name)
            for b in (1, 2, 7, 9):
                rows = [slice(i, i + b) for i in range(0, 36, b)]
                margins = np.vstack([ote.predict_margin(X.iloc[r]) for r in rows])
                index = np.vstack([ote.leaf_index(X.iloc[r]) for r in rows])
                assert np.array_equal(margins, base[:len(margins)]), (name, leaf_type, b)
                assert np.array_equal(index, leaves[:len(index)]), (name, leaf_type, b)
