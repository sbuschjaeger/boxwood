"""Tests for the pure-Python CatBoost prediction reference (``catboost_ref``).

Each test trains a tiny CatBoost model on synthetic data, saves it as JSON with
``pool=`` and checks that ``CatBoostRef`` reproduces ``predict(RawFormulaVal)``
within a relative tolerance. Test frames always contain category values that are
not in the training data and NaN in the numeric columns.
"""

import json

import numpy as np
import pandas as pd
import pytest
from catboost import Pool
from conftest import catboost_json, train_catboost

import boxwood as bw
from catboost_ref import CatBoostRef, cat_hash

TOL = 1e-5
CATS = ["c1", "c2", "c3", "c4"]


def make_data(n=3000, seed=2):
    """Float + categorical columns whose target needs combinations to fit."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "c1": rng.choice(list("abcdefg"), n),
        "c2": rng.choice(["x", "y", "z"], n),
        "c3": rng.integers(0, 40, n).astype(str),
        "c4": rng.integers(0, 30, n).astype(str),
    })
    y = (((X.c3.astype(int) + X.c4.astype(int)) % 5 == 0)
         ^ (X.c2 == "x")
         ^ ((X.c1 < "d") & (X.a > 0))).astype(int)
    return X, y


def train_save(tmp_path, X, y, name, cats=CATS, **params):
    """Train a tiny model, save it as JSON with the training pool; return (path, model)."""
    model = train_catboost(X, y, cats, **params)
    path = tmp_path / f"{name}.json"
    path.write_text(catboost_json(model, Pool(X, label=y, cat_features=cats)), encoding="utf-8")
    return path, model


def with_unseen(X, cat_cols=CATS):
    """The observations of X plus observations with unseen categories and NaN in numeric columns."""
    Xt = X.iloc[:200].copy().reset_index(drop=True)
    unseen = Xt.iloc[[0, 1]].copy()
    for c in cat_cols:
        unseen[c] = "unseen_zzz"
    Xt = pd.concat([Xt, unseen], ignore_index=True)
    for c in Xt.columns:
        if pd.api.types.is_numeric_dtype(Xt[c]):
            Xt.loc[1, c] = np.nan
            Xt.loc[len(Xt) - 1, c] = np.nan
    return Xt


def assert_close(got, expected):
    """|a - b| <= TOL * max(1, |b|), elementwise."""
    rel = np.abs(got - expected) / np.maximum(1.0, np.abs(expected))
    assert rel.max() <= TOL, f"max relative error {rel.max():.3e}"


def assert_margins(path, model, X_test):
    """CatBoostRef and boxwood margins must equal CatBoost's RawFormulaVal within TOL.

    The oracle is the trained model, not the model loaded back from the JSON file:
    CatBoost 1.2.10 loads some JSON CTR tables wrongly (multiclass
    BinarizedTargetMeanValue predicts differently after a JSON round trip).
    """
    expected = model.predict(X_test, prediction_type="RawFormulaVal").reshape(len(X_test), -1)
    got = CatBoostRef(json.loads(path.read_text())).margin(X_test)
    assert got.shape == expected.shape
    assert_close(got, expected)
    bw_got = bw.ObliviousTreeEnsemble.from_catboost(model).predict_margin(X_test)
    assert bw_got.shape == expected.shape
    assert_close(bw_got, expected)


def test_cat_hash_matches_model(tmp_path):
    """cat_hash equals the model's cat_features_hash for strings of length 0..70 and a few
    long ones (the CityHash loop over 64-byte blocks runs several times)."""
    rng = np.random.default_rng(11)
    alphabet = list("abcdefghij0123456789GrüßeÄÖ")
    strings = []
    for length in list(range(0, 71)) + [127, 128, 129, 200, 300]:
        for _ in range(2):
            strings.append("".join(str(rng.choice(alphabet)) for _ in range(length)))
    strings += ["", "0", "Grüße"]
    strings = list(dict.fromkeys(strings))
    X = pd.DataFrame({"c": strings})
    y = np.array([i % 2 for i in range(len(strings))])
    path, model = train_save(tmp_path, X, y, "hash", cats=["c"], iterations=10, depth=2)
    entries = json.loads(path.read_text())["features_info"]["cat_features_hash"]
    assert {len(e["value"]) for e in entries} >= set(range(0, 71))
    assert len(entries) == len(strings)
    for e in entries:
        assert cat_hash(e["value"]) == e["hash"], e["value"]


def test_binary_with_combinations(tmp_path):
    X, y = make_data()
    path, model = train_save(tmp_path, X, y, "binary", loss_function="Logloss", max_ctr_complexity=4)
    ctrs = json.loads(path.read_text())["features_info"]["ctrs"]
    assert any(len(c["elements"]) > 1 for c in ctrs), "expected combination CTRs"
    assert_margins(path, model, with_unseen(X))


def test_regression(tmp_path):
    X, _ = make_data()
    y = (X.a + (X.c3.astype(int) + X.c4.astype(int)) % 5
         + (X.c2 == "x") * 3.0).to_numpy()
    path, model = train_save(tmp_path, X, y, "reg", loss_function="RMSE", max_ctr_complexity=4)
    assert_margins(path, model, with_unseen(X))


def test_multiclass(tmp_path):
    X, _ = make_data()
    y = ((X.c3.astype(int) + X.c4.astype(int)) % 3).to_numpy()
    path, model = train_save(tmp_path, X, y, "mc", loss_function="MultiClass", classes_count=3,
                     max_ctr_complexity=4)
    assert_margins(path, model, with_unseen(X))


def test_one_hot(tmp_path):
    X, y = make_data()
    path, model = train_save(tmp_path, X, y, "onehot", one_hot_max_size=4, max_ctr_complexity=4)
    info = json.loads(path.read_text())["features_info"]
    assert any(c.get("values") is not None for c in info["categorical_features"]), "expected a one-hot feature"
    assert any(e["combination_element"] == "cat_feature_exact_value"
               for c in info["ctrs"] for e in c["elements"]), "expected a one-hot value in a combination"
    assert_margins(path, model, with_unseen(X))


CTR_TYPES = ["Borders", "Buckets", "BinarizedTargetMeanValue", "Counter",
             "FloatTargetMeanValue", "FeatureFreq"]

@pytest.mark.parametrize("loss", ["Logloss", "RMSE", "MultiClass"])
@pytest.mark.parametrize("ctr_type", CTR_TYPES)
def test_single_ctr_type(tmp_path, ctr_type, loss):
    X, y = make_data()
    if loss == "MultiClass":
        y = (X.c3.astype(int) + X.c4.astype(int)) % 3
    try:
        path, model = train_save(tmp_path, X, y, f"ctr_{ctr_type}_{loss}", loss_function=loss,
                         simple_ctr=[ctr_type], combinations_ctr=[ctr_type], max_ctr_complexity=4)
    except Exception as exc:  # CatBoost refuses some CTR types on CPU
        pytest.skip(f"{ctr_type} refused on CPU: {exc}")
    ctrs = json.loads(path.read_text())["features_info"]["ctrs"]
    assert {c["ctr_type"] for c in ctrs} == {ctr_type}
    assert_margins(path, model, with_unseen(X))


@pytest.mark.parametrize("ctr", ["Counter:Prior=0.5/1", "Borders:Prior=0.3/1"])
def test_ctr_prior_for_unseen_values(tmp_path, ctr):
    """Unseen values get the CTR of empty statistics; a nonzero prior makes it visible."""
    X, y = make_data()
    path, model = train_save(tmp_path, X, y, "prior", loss_function="Logloss",
                             simple_ctr=[ctr], combinations_ctr=[ctr], max_ctr_complexity=4)
    assert_margins(path, model, with_unseen(X))


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_nan_mode(tmp_path, nan_mode):
    X, y = make_data()
    X = X.copy()
    X.loc[::7, "a"] = np.nan
    path, model = train_save(tmp_path, X, y, f"nan_{nan_mode}", nan_mode=nan_mode)
    treatment = json.loads(path.read_text())["features_info"]["float_features"][0]["nan_value_treatment"]
    assert treatment == {"Min": "AsFalse", "Max": "AsTrue"}[nan_mode]
    assert_margins(path, model, with_unseen(X))


def test_depth_zero_trees(tmp_path):
    rng = np.random.default_rng(5)
    n = 1000
    X = pd.DataFrame({"k": np.ones(n), "c1": rng.choice(list("abc"), n)})
    y = (rng.random(n) > 0.5).astype(int)
    path, model = train_save(tmp_path, X, y, "depth0", cats=["c1"], iterations=20)
    trees = json.loads(path.read_text())["oblivious_trees"]
    assert any(not t.get("splits") for t in trees), "expected a depth-0 tree"
    assert_margins(path, model, with_unseen(X, cat_cols=["c1"]))
