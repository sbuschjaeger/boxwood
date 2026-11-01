"""Tests for the sklearn estimators ``BoxwoodClassifier`` and ``BoxwoodRegressor``.

All models are tiny (100 iterations, 2 threads, at most 3000 observations). The tests
check the estimator plumbing -- targets, holdout handling, attributes -- not
prediction quality.
"""

import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

import boxwood as bw
from test_catboost_ref import make_data

CB = {"iterations": 100, "thread_count": 2}


def test_binary_string_labels():
    X, y = make_data(n=1000, seed=2)
    labels = np.where(y == 1, "yes", "no")
    clf = bw.BoxwoodClassifier(catboost_params=CB).fit(X, labels)
    assert list(clf.classes_) == ["no", "yes"]
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert_allclose(proba.sum(axis=1), 1.0)
    margin = clf.ensemble_.predict_margin(X)[:, 0]
    assert_allclose(proba[:, 1], 1.0 / (1.0 + np.exp(-margin)))
    assert_array_equal(clf.predict(X), np.where(proba[:, 1] > 0.5, "yes", "no"))


def test_multiclass_and_regressor():
    X, _ = make_data(n=1200, seed=3)
    y3 = np.digitize(X["a"], [-0.5, 0.5])
    clf = bw.BoxwoodClassifier(catboost_params=CB).fit(X, y3)
    assert clf.predict_proba(X).shape == (len(X), 3)
    assert clf.predict(X).shape == (len(X),)
    assert np.isfinite(clf.score(X, y3))

    y = X["a"].to_numpy()
    reg = bw.BoxwoodRegressor(catboost_params=CB).fit(X, y)
    pred = reg.predict(X)
    assert pred.shape == (len(X),)
    assert_allclose(pred, reg.ensemble_.predict_margin(X)[:, 0])
    assert np.isfinite(reg.score(X, y))


def test_holdout_given_vs_split():
    X, y = make_data(n=1000, seed=4)
    split = bw.BoxwoodClassifier(catboost_params=CB).fit(X, y)
    assert split.fit_report_ is not None
    assert np.isfinite(split.ensemble_.info()["fit_lambda"])

    given = bw.BoxwoodClassifier(catboost_params=CB).fit(
        X, y, X_holdout=X.iloc[:200], y_holdout=y.iloc[:200])
    assert given.fit_report_ is not None
    assert np.isfinite(given.ensemble_.info()["fit_lambda"])


def test_clone_get_params_pickle():
    X, y = make_data(n=800, seed=7)
    clf = bw.BoxwoodClassifier(catboost_params=CB, strategy="path", random_state=3)
    clf.fit(X, y)
    params = clf.get_params()
    assert params["strategy"] == "path"
    assert params["random_state"] == 3
    assert params["catboost_params"] == CB
    fresh = clone(clf)
    assert fresh.get_params() == params
    assert not hasattr(fresh, "ensemble_")
    again = pickle.loads(pickle.dumps(clf))
    assert_allclose(again.predict_proba(X), clf.predict_proba(X))


def test_numpy_input_matches_frame():
    X, y = make_data(n=800, seed=8)
    clf_df = bw.BoxwoodClassifier(catboost_params=CB).fit(X, y)
    clf_np = bw.BoxwoodClassifier(catboost_params=CB).fit(X.to_numpy(), y)
    assert clf_np.n_features_in_ == clf_df.n_features_in_
    assert not hasattr(clf_np, "feature_names_in_")
    assert_array_equal(clf_df.feature_names_in_, np.asarray(X.columns))
    assert_allclose(clf_np.predict_proba(X), clf_df.predict_proba(X), rtol=1e-9, atol=1e-12)
