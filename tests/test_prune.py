"""Pruning to a split budget as part of ``fit``.

The tests train tiny CatBoost models (numeric and with categorical features),
prune the imported ensemble with ``split_budget`` and check the budget, the
prefix property of the cut trees, the level-basis truncation (cut leaves are
the means of the original leaves over the cut groups), the budget as a hard
limit also with a holdout, and the bias folding of removed trees.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from conftest import catboost_json, train_catboost

import boxwood as bw
from test_ensemble import assert_close, make_X, make_y


def _train_json(X, y, cat_cols, **params):
    """Train a tiny CatBoost RMSE model and return its JSON text."""
    return catboost_json(train_catboost(X, y, cat_cols, **{"iterations": 30, "loss_function": "RMSE", **params}))


@pytest.fixture(scope="module")
def num_model():
    """A numeric RMSE model: 300 observations, 30 iterations, depth 4."""
    X = make_X(n=300, seed=7)
    y = make_y("RMSE", X, seed=7)
    return _train_json(X, y, None), X, y


@pytest.fixture(scope="module")
def cat_model():
    """A model with categorical features (CatBoost builds CTRs for them)."""
    rng = np.random.default_rng(21)
    n = 300
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "c1": rng.choice(list("abcde"), n),
        "c2": rng.choice(["x", "y", "z"], n),
    })
    eff = X["c1"].map({"a": 1.5, "b": -1.0, "c": 0.5, "d": 0.0, "e": -1.5})
    y = (X["a"] + eff + 0.3 * rng.normal(size=n)).to_numpy()
    return _train_json(X, y, ["c1", "c2"]), X, y


def _stored_leaves(js, tree=0):
    """The tree's leaf values exactly as boxwood stores them (float32 of scale * v)."""
    obj = json.loads(js)
    scale = obj["scale_and_bias"][0]
    lv = scale * np.asarray(obj["oblivious_trees"][tree]["leaf_values"], dtype=np.float64)
    return lv.astype(np.float32).astype(np.float64)


def _bias(js):
    return json.loads(js)["scale_and_bias"][1][0]


# 1. The budget holds and the cut trees are prefixes of the original trees.


@pytest.mark.parametrize("model_name", ["num_model", "cat_model"])
@pytest.mark.parametrize("frac", [0.25, 0.5, 0.75])
def test_budget_respected_and_prefixes(request, model_name, frac):
    js, X, y = request.getfixturevalue(model_name)
    full = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    total = full.info()["n_splits"]
    orig = full.splits()
    B = max(1, int(total * frac))
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    rep = ote.fit(X, y, strategy="fixed", lam=1.0, max_iter=5, split_budget=B)
    assert rep["splits_used"] <= B
    pruned = ote.splits()
    assert rep["splits_used"] == sum(len(t) for t in pruned)
    assert len(orig) - len(pruned) == rep["trees_removed"]
    # The kept trees appear in the original order, each truncated to a prefix.
    oi = 0
    for pt in pruned:
        while oi < len(orig) and pt != orig[oi][:len(pt)]:
            oi += 1
        assert oi < len(orig), "a pruned tree is not a prefix of a later original tree"
        oi += 1



@pytest.mark.parametrize("model_name", ["num_model", "cat_model"])
@pytest.mark.parametrize("leaf_type", ["float32", "int8"])
@pytest.mark.parametrize("frac", [0.6, 0.8])  # less leaves no room for trees in the small int8 models
def test_max_bytes_respected(request, model_name, leaf_type, frac):
    js, X, y = request.getfixturevalue(model_name)
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js, leaf_type)
    size = len(ote.to_bytes())
    B = int(size * frac)
    rep = ote.fit(X, y, strategy="fixed", lam=1.0, max_iter=5, max_bytes=B)
    assert len(ote.to_bytes()) <= B
    assert rep["splits_used"] < sum(len(t) for t in bw.ObliviousTreeEnsemble.from_catboost_json(js).splits())
    assert ote.info()["leaf_type"] == leaf_type


@pytest.mark.parametrize("leaf_type", ["float32", "int8"])
def test_max_bytes_of_the_model_keeps_it(num_model, leaf_type):
    # The cost of the trees is exact: a budget of the model's own size keeps every split, and the
    # refit model has that size to the byte.
    js, X, y = num_model
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js, leaf_type)
    size, total = len(ote.to_bytes()), ote.info()["n_splits"]
    rep = ote.fit(X, y, strategy="fixed", lam=1e12, max_iter=0, max_bytes=size)
    assert rep["splits_used"] == total
    assert len(ote.to_bytes()) == size
    # Removing all trees leaves the rest: the trees took tree_bytes of the model.
    trees = ote.info()["tree_bytes"]
    ote.fit(X, y, strategy="fixed", lam=1e12, max_iter=0, max_bytes=size - trees)
    assert ote.info()["n_trees"] == 0
    assert len(ote.to_bytes()) == size - trees


def test_max_bytes_errors(num_model):
    js, X, y = num_model
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    with pytest.raises(ValueError, match="not both"):
        ote.fit(X, y, strategy="fixed", split_budget=5, max_bytes=10**6)
    with pytest.raises(ValueError, match="without trees"):
        ote.fit(X, y, strategy="fixed", max_bytes=10)

# 2. A budget of all splits with a frozen leaf fit leaves the model unchanged.


def test_full_budget_identity(num_model):
    js, X, y = num_model
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    total = ote.info()["n_splits"]
    before = ote.predict_margin(X)
    rep = ote.fit(X, y, strategy="fixed", lam=1e12, max_iter=0, split_budget=total)
    assert rep["splits_used"] == total
    assert rep["trees_removed"] == 0
    assert_close(ote.predict_margin(X), before, tol=1e-5)


# 3. Cutting one tree to k levels gives the group means of its leaves.
#
# The model is hand-made (no training): its leaves make the level-basis g concave
# at the top, so a budget of 1 keeps exactly the root level.


@pytest.fixture(scope="module")
def one_tree():
    """A single depth-3 tree from a trained tiny CatBoost model."""
    X = make_X(n=300, seed=7)
    y = make_y("RMSE", X, seed=7)
    return _train_json(X, y, None,
                      iterations=1, depth=3), X, y


# One depth-3 tree; the leaf values give g = (0, 13, 15, 35), whose upper
# concave hull starts with a segment of length 1, so B = 1 keeps one level.
SINGLE_TREE_JS = (
    '{"model_info":{"params":{"loss_function":{"type":"RMSE"}}},'
    '"scale_and_bias":[1.0,[0.1]],'
    '"features_info":{"float_features":['
    '{"flat_feature_index":0,"borders":[0.0,1.0],"nan_value_treatment":"AsFalse"},'
    '{"flat_feature_index":1,"borders":[0.0],"nan_value_treatment":"AsFalse"}]},'
    '"oblivious_trees":[{"splits":['
    '{"float_feature_index":0,"border":0.0,"split_type":"FloatFeature"},'
    '{"float_feature_index":1,"border":0.0,"split_type":"FloatFeature"},'
    '{"float_feature_index":0,"border":1.0,"split_type":"FloatFeature"}],'
    '"leaf_values":[10,0,8,2,6,2,4,0]}]}'
)


def test_cut_leaves_are_group_means():
    ref = bw.ObliviousTreeEnsemble.from_catboost_json(SINGLE_TREE_JS)
    d = ref.info()["n_splits"]
    assert d == 3
    X = np.random.default_rng(5).uniform(-2.0, 2.0, size=(200, 2)).astype(np.float32)
    y = np.zeros(len(X))  # unused: the leaf fit runs zero iterations
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(SINGLE_TREE_JS)
    rep = ote.fit(X, y, strategy="fixed", lam=1e12, max_iter=0, split_budget=1)
    k = ote.info()["n_splits"]
    assert 0 < k < d, f"B=1 kept k={k}, expected a middle depth"
    assert rep["trees_removed"] == 0
    leaves = _stored_leaves(SINGLE_TREE_JS)
    bias = _bias(SINGLE_TREE_JS)
    leaf_orig = ref.leaf_index(X)[:, 0]
    leaf_new = ote.leaf_index(X)[:, 0]
    assert np.array_equal(leaf_new, leaf_orig % (1 << k))
    m = ote.predict_margin(X)[:, 0]
    for p in range(1 << k):
        in_group = np.flatnonzero(leaf_orig % (1 << k) == p)
        assert in_group.size > 0, f"cut group {p} has no observations"
        # The cut leaf is the plain mean of all its children, also of leaves no
        # observation reaches (here leaves 4 and 6: x0 <= 0 and x0 > 1).
        expected = bias + leaves[np.arange(leaves.size) % (1 << k) == p].mean()
        assert np.allclose(m[in_group], expected, rtol=1e-5, atol=1e-5)


# 4. The budget is a hard limit: with a holdout, a pruned model that is clearly
# worse than the unpruned one is kept.


def test_budget_is_hard_with_holdout(num_model):
    js, X, y = num_model
    Xh = make_X(n=150, seed=99)
    yh = make_y("RMSE", Xh, seed=99)
    ote_p = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    rep_p = ote_p.fit(X, y, Xh, yh, strategy="path", max_iter=10, split_budget=1)
    assert rep_p["splits_used"] <= 1
    assert ote_p.info()["n_splits"] == rep_p["splits_used"]
    ote_u = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    rep_u = ote_u.fit(X, y, Xh, yh, strategy="path", max_iter=10)
    assert rep_p["holdout_loss"] > rep_u["holdout_loss"]


# 5. A removed tree is folded into the bias.


def test_removed_tree_folds_into_bias(one_tree):
    js, X, y = one_tree
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    rep = ote.fit(X, y, strategy="fixed", lam=1e12, max_iter=0, split_budget=0)
    assert rep["trees_removed"] == 1
    assert ote.info()["n_trees"] == 0
    expected = _bias(js) + _stored_leaves(js).mean()
    m = ote.predict_margin(X)[:, 0]
    assert np.allclose(m, expected, rtol=1e-5, atol=1e-5)


# 6. Pickle round trip of a pruned ensemble; the estimators accept split_budget.


@pytest.mark.parametrize("model_name", ["num_model", "cat_model"])
def test_pickle_round_trip(request, model_name):
    js, X, y = request.getfixturevalue(model_name)
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    rep = ote.fit(X, y, strategy="fixed", lam=1.0, max_iter=3, split_budget=10)
    assert rep["splits_used"] <= 10
    data = ote.to_bytes()
    back = bw.ObliviousTreeEnsemble.from_bytes(data)
    assert back.to_bytes() == data
    assert np.array_equal(back.predict_margin(X), ote.predict_margin(X))
    again = pickle.loads(pickle.dumps(ote))
    assert again.to_bytes() == data
    assert np.array_equal(again.predict_margin(X), ote.predict_margin(X))


def test_estimator_split_budget():
    rng = np.random.default_rng(31)
    n = 400
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c1": rng.choice(list("abcd"), n),
    })
    y = (X["a"] - X["b"] + X["c1"].map({"a": 1.0, "b": 0.0, "c": -1.0, "d": 0.5})).to_numpy()
    est = bw.BoxwoodRegressor(
        catboost_params={"iterations": 30, "depth": 4, "thread_count": 2}, split_budget=15)
    est.fit(X, y)
    rep = est.fit_report_
    assert rep["splits_used"] <= 15
    assert est.ensemble_.info()["n_splits"] == rep["splits_used"]
    assert est.get_params()["split_budget"] == 15


def test_estimator_max_bytes():
    rng = np.random.default_rng(32)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] - X["b"]).to_numpy()
    params = {"iterations": 60, "depth": 4, "thread_count": 2}
    full = bw.BoxwoodRegressor(catboost_params=params).fit(X, y)
    B = len(full.ensemble_.to_bytes()) // 2
    est = bw.BoxwoodRegressor(catboost_params=params, max_bytes=B).fit(X, y)
    assert len(est.ensemble_.to_bytes()) <= B
    assert est.ensemble_.info()["leaf_type"] == "int8"
