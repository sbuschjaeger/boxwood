"""Tests for the Python ensemble around the native core.

The oracle is the trained CatBoost model itself, never a model loaded back from
JSON: CatBoost 1.2.10 misreads some of its own JSON tables, so only the
in-memory model can be trusted. Every case feeds the ensemble observations that hold
NaN, +/-inf, values exactly on a border and values outside all borders.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from conftest import catboost_json, train_catboost

import boxwood as bw
from test_catboost_ref import make_data, train_save, with_unseen

TOL = 1e-5
CONST_COL = 2  # a column of ones: no border, therefore no tree uses it


def make_X(n=2000, seed=7):
    """Numeric features on several scales, with a constant column at CONST_COL."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    X[:, 1] = X[:, 1] * 10.0 - 5.0
    X[:, 2] = 1.0
    X[:, 3] = np.exp(X[:, 3])
    return X


def make_y(loss, X, seed=7):
    """A target that fits the loss: labels, probabilities or one column per target."""
    rng = np.random.default_rng(seed + 1)
    score = X[:, 0] + 0.1 * X[:, 1] - 0.5 * np.log1p(X[:, 3])
    if loss == "RMSE":
        return score + 0.1 * rng.normal(size=len(X))
    if loss == "Logloss":
        return (score + 0.3 * rng.normal(size=len(X)) > 0).astype(float)
    if loss == "CrossEntropy":
        return 1.0 / (1.0 + np.exp(-score))
    if loss == "MultiClass":
        return np.digitize(score, [-0.5, 0.0, 0.5])
    raise ValueError(loss)


def probe_observations(X_train, model_json):
    """Rows covering exact borders, values outside all borders, NaN and +/-inf.

    Each probe changes one column of a training observation, so every border and every
    extreme value is tested on its own.
    """
    feats = model_json["features_info"].get("float_features", [])
    xs = [X_train[:20]]
    base = X_train[0].astype(np.float32)
    for j, f in enumerate(feats):
        borders = f.get("borders") or []
        probes = []
        if borders:
            lo, hi = np.float32(borders[0]), np.float32(borders[-1])
            probes += [lo, hi, lo - np.float32(1.0), hi + np.float32(1.0)]
        probes += [np.float32(-np.inf), np.float32(np.inf), np.float32(np.nan)]
        for v in probes:
            x = base.copy()
            x[j] = v
            xs.append(x)
    Xt = np.vstack(xs).astype(np.float32)
    assert np.isnan(Xt).any() and np.isinf(Xt).any()
    return Xt


def assert_close(a, b, tol=TOL):
    """|a - b| <= tol * max(1, |b|), elementwise."""
    rel = np.abs(a - b) / np.maximum(1.0, np.abs(b))
    assert rel.max() <= tol, f"max relative error {rel.max():.3e}"


def check_margins(model, ote, Xt):
    """boxwood margins must equal CatBoost's RawFormulaVal."""
    K = ote.info()["n_outputs"]
    got = ote.predict_margin(Xt)
    exp = model.predict(Xt, prediction_type="RawFormulaVal").reshape(len(Xt), K)
    assert got.shape == (len(Xt), K)
    assert_close(got, exp)


def leaf_index_from_splits(ote, model_json, Xt):
    """Recompute the leaf index of every observation for every tree from splits() + borders.

    Only numeric splits are handled here; the categorical cases are covered by
    the C++ tests and the categorical Python tests.
    """
    feats = model_json["features_info"].get("float_features", [])
    out = np.zeros((len(Xt), len(model_json["oblivious_trees"])), dtype=np.uint8)
    for t, tree_splits in enumerate(ote.splits()):
        leaf = np.zeros(len(Xt), dtype=np.uint8)
        for level, (source, col, kind, arg) in enumerate(tree_splits):
            assert source == "numeric", f"numeric-only test, got source {source!r}"
            borders = np.asarray(feats[col]["borders"], dtype=np.float32)
            x = Xt[:, col]
            # The code counts the borders below x; NaN gets all of them or none.
            code = (x[:, None] > borders[None, :]).sum(axis=1)
            nan_high = feats[col]["nan_value_treatment"] == "AsTrue"
            code = np.where(np.isnan(x), len(borders) if nan_high else 0, code)
            bit = code > arg if kind == "gt" else code == arg
            leaf |= bit.astype(np.uint8) << level
        out[:, t] = leaf
    return out


@pytest.fixture(scope="session")
def models():
    """Train each named model at most once per session.

    Returns a getter ``get(name, X, y, **params)`` giving ``(model, ote, json)``
    where ``ote`` is the ensemble imported from the model's JSON and ``model`` is
    the trained CatBoost model the tests compare against.
    """
    cache = {}

    def get(name, X, y, **params):
        if name not in cache:
            model = train_catboost(X, y, **{"depth": 6, **params})
            text = catboost_json(model)
            ote = bw.ObliviousTreeEnsemble.from_catboost_json(text)
            # from_catboost must reach the same model as from_catboost_json.
            assert bw.ObliviousTreeEnsemble.from_catboost(model).to_bytes() == ote.to_bytes()
            cache[name] = (model, ote, json.loads(text))
        return cache[name]

    return get


@pytest.mark.parametrize(
    "loss,task",
    [("RMSE", "regression"), ("Logloss", "binary"), ("CrossEntropy", "binary"),
     ("MultiClass", "multiclass")],
)
def test_losses(models, loss, task):
    X = make_X()
    model, ote, js = models(loss, X, make_y(loss, X), loss_function=loss)
    assert ote.info()["task"] == task
    check_margins(model, ote, probe_observations(X, js))


@pytest.mark.parametrize("depth", [1, 6, 8])
def test_depths(models, depth):
    X = make_X()
    model, ote, js = models(f"depth{depth}", X, make_y("RMSE", X),
                            loss_function="RMSE", depth=depth, iterations=300)
    assert ote.info()["n_trees"] == 300  # more than the 64-tree block
    # A level with no usable border stops the tree early, so trees are at most `depth`.
    assert all(len(t) <= depth for t in ote.splits())
    assert any(len(t) == depth for t in ote.splits())
    check_margins(model, ote, probe_observations(X, js))


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_nan_mode(models, nan_mode):
    X = make_X()
    y = make_y("RMSE", X)
    X[::7, 1] = np.nan  # NaN in the training data, kept out of the target
    model, ote, js = models(f"nan_{nan_mode}", X, y, loss_function="RMSE", nan_mode=nan_mode)
    treatment = js["features_info"]["float_features"][1]["nan_value_treatment"]
    assert treatment == {"Min": "AsFalse", "Max": "AsTrue"}[nan_mode]
    check_margins(model, ote, probe_observations(X, js))


def test_depth_zero_trees():
    """A depth-0 tree is one leaf and no splits; it shifts every margin alike.

    A trained numeric-only CatBoost model always finds a split, so the depth-0
    trees are taken from a model JSON instead (the same shape CatBoost writes
    when a tree has nothing to split on).
    """
    js = {
        "model_info": {"params": {"loss_function": {"type": "RMSE"}}},
        "scale_and_bias": [2.0, [0.5]],
        "features_info": {
            "float_features": [
                {"flat_feature_index": 0, "borders": [0.0], "nan_value_treatment": "AsFalse"}
            ]
        },
        "oblivious_trees": [
            {"splits": None, "leaf_values": [1.0]},
            {"splits": [{"float_feature_index": 0, "border": 0.0, "split_type": "FloatFeature"}],
             "leaf_values": [2.0, 3.0]},
            {"splits": [], "leaf_values": [0.25]},
        ],
    }
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(json.dumps(js))
    assert [len(t) for t in ote.splits()] == [0, 1, 0]
    X = np.array([[-1.0], [0.0], [1.0], [np.nan]], dtype=np.float32)
    assert ote.leaf_index(X).tolist() == [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]]
    # bias + scale * (1 + leaf of the split tree + 0.25); NaN is below the border
    expected = 0.5 + 2.0 * (1.25 + np.where(X[:, 0] > 0, 3.0, 2.0))
    assert_close(ote.predict_margin(X).ravel(), expected)
    assert np.array_equal(
        bw.ObliviousTreeEnsemble.from_bytes(ote.to_bytes()).predict_margin(X),
        ote.predict_margin(X),
    )


def test_unused_column(models):
    X = make_X()
    model, ote, js = models("unused", X, make_y("RMSE", X), loss_function="RMSE")
    used = {s[1] for tree in ote.splits() for s in tree}
    assert CONST_COL not in used, "the constant column must not be split on"
    assert ote.info()["n_numeric"] == X.shape[1]
    check_margins(model, ote, probe_observations(X, js))


@pytest.mark.parametrize(
    "loss,kind",
    [("Logloss", "binary"), ("MultiClass", "multiclass"), ("RMSE", "regression")],
)
def test_predict(models, loss, kind):
    X = make_X()
    model, ote, js = models(f"pred_{loss}", X, make_y(loss, X), loss_function=loss)
    assert ote.info()["task"] == kind
    Xt = probe_observations(X, js)
    p = ote.predict(Xt)
    if kind == "binary":
        assert_close(p[:, 0], model.predict(Xt, prediction_type="Probability")[:, 1])
    elif kind == "multiclass":
        assert_close(p, model.predict(Xt, prediction_type="Probability"))
    else:
        assert_close(p, model.predict(Xt, prediction_type="RawFormulaVal").reshape(p.shape))


def test_leaf_index_matches_splits(models):
    X = make_X()
    model, ote, js = models("leafsplits", X, make_y("RMSE", X), loss_function="RMSE")
    Xt = probe_observations(X, js)
    assert np.array_equal(leaf_index_from_splits(ote, js, Xt), ote.leaf_index(Xt))
    assert ote.leaf_index(Xt).shape == (len(Xt), ote.info()["n_trees"])
    assert ote.leaf_index(Xt).dtype == np.uint8
    check_margins(model, ote, Xt)


def test_leaves_and_bias_give_the_margin(models):
    X = make_X()
    _, ote, js = models("MultiClass", X, make_y("MultiClass", X), loss_function="MultiClass")
    Xt = probe_observations(X, js)
    idx = ote.leaf_index(Xt)
    # The prediction adds the leaves to a float64 margin in tree order, so the
    # same sum in numpy gives the same bits.
    margin = np.tile(ote.bias(), (len(Xt), 1))
    for t, leaves in enumerate(ote.leaves()):
        assert leaves.shape == (2 ** len(ote.splits()[t]), ote.info()["n_outputs"])
        margin += leaves[idx[:, t]]
    assert np.array_equal(margin, ote.predict_margin(Xt))


def test_pickle_round_trip(models):
    X = make_X()
    _, ote, js = models("pickle", X, make_y("RMSE", X), loss_function="RMSE")
    Xt = probe_observations(X, js)
    data = ote.to_bytes()
    assert bw.ObliviousTreeEnsemble.from_bytes(data).to_bytes() == data
    back = pickle.loads(pickle.dumps(ote))
    assert back.to_bytes() == data
    assert np.array_equal(back.predict_margin(Xt), ote.predict_margin(Xt))
    assert np.array_equal(back.leaf_index(Xt), ote.leaf_index(Xt))


def test_accepts_dataframe(models):
    X = make_X(n=500)
    _, ote, js = models("frame", X, make_y("RMSE", X), loss_function="RMSE")
    Xt = probe_observations(X, js)
    frame = pd.DataFrame(Xt, columns=[f"f{i}" for i in range(Xt.shape[1])])
    assert np.array_equal(ote.predict_margin(frame), ote.predict_margin(Xt))


@pytest.mark.parametrize("bad", ["extra_columns", "missing_columns", "one_d", "truncated"])
def test_errors(models, bad):
    X = make_X(n=500)
    _, ote, _ = models("errors", X, make_y("RMSE", X), loss_function="RMSE")
    if bad == "extra_columns":
        with pytest.raises(ValueError):
            ote.predict(np.zeros((5, X.shape[1] + 1), dtype=np.float32))
    elif bad == "missing_columns":
        with pytest.raises(ValueError):
            ote.predict_margin(np.zeros((5, X.shape[1] - 1), dtype=np.float32))
    elif bad == "one_d":
        with pytest.raises(ValueError):
            ote.leaf_index(np.zeros(X.shape[1], dtype=np.float32))
    else:
        with pytest.raises(ValueError):
            bw.ObliviousTreeEnsemble.from_bytes(ote.to_bytes()[:-1])


def test_error_is_value_error():
    assert issubclass(bw.Error, ValueError)


# ------------------------------------------------------------- categorical
# These use the generators and the tiny-training helper of test_catboost_ref.


def test_categorical_interleaved(tmp_path):
    """Categorical columns interleaved with numeric ones: c1, a, c2, c3."""
    X, y = make_data(n=1000)
    X = X[["c1", "a", "c2", "c3"]]
    _, model = train_save(tmp_path, X, y, "interleaved", cats=["c1", "c2"])
    ote = bw.ObliviousTreeEnsemble.from_catboost(model)
    assert ote.info()["numeric_flat_index"] == [1, 3]
    assert ote.info()["categorical_flat_index"] == [0, 2]
    check_margins(model, ote, with_unseen(X, cat_cols=["c1", "c2"]))


def test_categorical_input_forms(tmp_path):
    """The same observations as a DataFrame and as a numpy object array give the same margins."""
    X, y = make_data(n=1000)
    X = X[["c1", "a", "c2"]]
    _, model = train_save(tmp_path, X, y, "inputforms", cats=["c1", "c2"])
    ote = bw.ObliviousTreeEnsemble.from_catboost(model)
    Xt = with_unseen(X, cat_cols=["c1", "c2"])
    check_margins(model, ote, Xt)
    obj = Xt.to_numpy(dtype=object)
    got = ote.predict_margin(obj)
    exp = model.predict(Xt, prediction_type="RawFormulaVal").reshape(got.shape)
    assert got.shape == exp.shape
    assert_close(got, exp)


def test_integer_categories(tmp_path):
    """CatBoost trained on an int column given as cat_feature; boxwood gets the ints too."""
    X, y = make_data(n=1000)
    X = X[["c3", "a"]].copy()
    X["c3"] = X["c3"].astype(int)
    _, model = train_save(tmp_path, X, y, "intcat", cats=["c3"])
    ote = bw.ObliviousTreeEnsemble.from_catboost(model)
    check_margins(model, ote, X)
    got = ote.predict_margin(X.to_numpy(dtype=object))
    exp = model.predict(X, prediction_type="RawFormulaVal").reshape(got.shape)
    assert got.shape == exp.shape
    assert_close(got, exp)
    check_margins(model, ote, with_unseen(X, cat_cols=["c3"]))


def test_pickle_categorical_round_trip(tmp_path):
    """A categorical model survives pickling: same bytes, bit-identical margins."""
    X, y = make_data(n=1000)
    _, model = train_save(tmp_path, X, y, "catpickle")
    ote = bw.ObliviousTreeEnsemble.from_catboost(model)
    data = ote.to_bytes()
    assert bw.ObliviousTreeEnsemble.from_bytes(data).to_bytes() == data
    back = pickle.loads(pickle.dumps(ote))
    assert back.to_bytes() == data
    Xt = with_unseen(X)
    assert np.array_equal(back.predict_margin(Xt), ote.predict_margin(Xt))


def test_categorical_small_batches(tmp_path):
    """Small batches (columns passed as they are) give the margins of one large batch (dictionary-encoded)."""
    X, y = make_data(n=1000)
    _, model = train_save(tmp_path, X, y, "smallbatch")
    ote = bw.ObliviousTreeEnsemble.from_catboost(model)
    Xt = pd.concat([with_unseen(X), X.iloc[200:500]], ignore_index=True)  # 502 observations
    full = ote.predict_margin(Xt)
    nullable = Xt.astype({"a": "Float64"})  # its NaN become pd.NA
    for frame in (Xt, Xt.to_numpy(dtype=object), nullable):
        parts = [ote.predict_margin(frame[i:i + 10]) for i in range(0, len(Xt), 10)]
        assert np.array_equal(np.vstack(parts), full)
    assert np.array_equal(ote.predict_margin(nullable), full)


def test_category_strings():
    assert bw.category_strings([1, "a", None, np.nan, pd.NA, 2.5]) == [
        "1", "a", "nan", "nan", "nan", "2.5",
    ]


def test_categorical_wrong_columns(tmp_path):
    X, y = make_data(n=500)
    X = X[["c1", "a"]]
    _, model = train_save(tmp_path, X, y, "catcols", cats=["c1"])  # 1 categorical + 1 numeric
    ote = bw.ObliviousTreeEnsemble.from_catboost(model)
    with pytest.raises(ValueError):
        ote.predict_margin(X.iloc[:, :1])
    with pytest.raises(ValueError):
        ote.predict(np.zeros((5, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        ote.leaf_index([1, 2, 3])


def test_categorical_splits(tmp_path):
    X, y = make_data(n=1000)
    _, model = train_save(tmp_path, X, y, "catsplits", one_hot_max_size=4, max_ctr_complexity=4)
    ote = bw.ObliviousTreeEnsemble.from_catboost(model)
    sources = {s[0] for tree in ote.splits() for s in tree}
    assert "onehot" in sources, ote.splits()
    assert "ctr" in sources, ote.splits()
