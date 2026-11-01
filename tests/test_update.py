"""Online-update tests: the update objective against a fixed fit, state
continuation across updates and pickles, and a stream with drift.

The oracle: after a fit, `update(X, y, steps=n)` on the ensemble and a
`fit(strategy="fixed", lam=fit_lambda, max_iter=n)` on a copy start from the
same leaves with the same sigma and optimizer, so their objectives agree.
"""

import pickle

import numpy as np
import pytest

import boxwood as bw
from test_ensemble import make_X, make_y
from test_fit import TASKS, _mc, target


@pytest.mark.parametrize("loss,task", TASKS)
def test_update_matches_fixed_fit(trained, loss, task):
    """3000 update steps reach the objective of a 3000-iteration fixed fit."""
    X = make_X(n=300, seed=5)
    y = target(loss, X, 5)
    Xh = make_X(n=120, seed=99)
    yh = target(loss, Xh, 99)
    js = trained(f"upd_{loss}", X, y, loss_function=loss, **_mc(loss))
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    ote.fit(X, y, Xh, yh, strategy="path", max_iter=50)
    lam = ote.info()["fit_lambda"]
    assert np.isfinite(lam)

    copy = bw.ObliviousTreeEnsemble.from_bytes(ote.to_bytes())
    ref = copy.fit(X, y, strategy="fixed", lam=lam, max_iter=3000, tol=1e-13)
    rep = ote.update(X, y, steps=3000)
    assert rep["steps"] == 3000
    assert abs(rep["objective"] - ref["objective"]) <= 1e-6 * max(1.0, abs(ref["objective"]))


def test_update_uses_stored_sigma(trained):
    """The update objective uses the stored fit sigma, not the window's."""
    X = make_X(n=300, seed=5)
    y = make_y("RMSE", X, seed=5)
    Xh = make_X(n=120, seed=99)
    yh = make_y("RMSE", Xh, seed=99)
    js = trained("upd_sig", X, y, loss_function="RMSE")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    ote.fit(X, y, Xh, yh, strategy="path", max_iter=50)
    sigma = ote.info()["fit_sigma"]
    assert sigma > 0

    X2 = make_X(n=200, seed=17)
    y2 = 4.0 * make_y("RMSE", X2, seed=17)  # a window with a different spread
    assert abs(np.std(y2) - sigma) > sigma

    margins = ote.predict_margin(X2)[:, 0] / sigma  # the anchor, before the update
    rep = ote.update(X2, y2, steps=1)
    J = 0.5 * np.mean((margins - y2 / sigma) ** 2)  # penalty 0 at the anchor
    # float32 leaves vs the float64 anchor give ~1e-8; the window's own sigma
    # would move J by a factor of ~16, so this pins the stored sigma.
    assert abs(rep["objective_start"] - J) <= 1e-6 * max(1.0, abs(J))


@pytest.mark.parametrize("optimizer", ["adam", "tree_newton"])
def test_pickle_continues_identically(trained, optimizer):
    """A pickled ensemble continues the update bit-identically."""
    X = make_X(n=200, seed=5)
    y = make_y("Logloss", X, seed=5)
    js = trained("upd_ser", X, y, loss_function="Logloss")
    A = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    A.fit(X, y, strategy="fixed", lam=1.0, optimizer=optimizer, max_iter=10)
    A.update(X[:100], y[:100], steps=2)
    B = pickle.loads(pickle.dumps(A))
    assert B.to_bytes() == A.to_bytes()
    A.update(X[100:], y[100:], steps=3)
    B.update(X[100:], y[100:], steps=3)
    assert A.to_bytes() == B.to_bytes()


def test_adam_state_continues(trained):
    """The Adam state continues across updates and is not reset to fresh.

    Note: two 1-step updates are NOT the same as one 2-step update -- each
    update re-anchors at the float32 leaves written back by the previous
    one, so the objective function itself changes between the updates.
    What continues is the optimizer state (t, m, v), verified here by
    pickle continuation and by the difference from a fresh Adam start.
    """
    X = make_X(n=200, seed=5)
    y = make_y("Logloss", X, seed=5)
    js = trained("upd_adam", X, y, loss_function="Logloss")

    def fitted():
        e = bw.ObliviousTreeEnsemble.from_catboost_json(js)
        e.fit(X, y, strategy="fixed", lam=1.0, optimizer="adam", learning_rate=0.05, max_iter=5)
        return e

    A = fitted()
    A.update(X, y, steps=1)  # t: 5 -> 6
    after_one = A.to_bytes()
    A.update(X, y, steps=1)  # t: 6 -> 7, from the stored state

    # The stored state is what the next update uses: continuing the mid-stream
    # ensemble reaches the same bytes as A.
    C = bw.ObliviousTreeEnsemble.from_bytes(after_one)
    C.update(X, y, steps=1)
    assert C.to_bytes() == A.to_bytes()
    B = pickle.loads(pickle.dumps(A))
    assert B.to_bytes() == A.to_bytes()

    # A fresh Adam start (t = 0 -> 1) from the same leaves is not the
    # continuation (t = 6 -> 7): the state was not reset.
    D = bw.ObliviousTreeEnsemble.from_bytes(after_one)
    D.fit(X, y, strategy="fixed", lam=1.0, optimizer="adam", learning_rate=0.05, max_iter=1)
    assert D.to_bytes() != A.to_bytes()


def test_stream_drift(trained):
    """On a stream with drift, one update step per window beats a frozen model."""
    n_train, n_win, win = 1000, 20, 100
    n = n_train + n_win * win
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    drift = np.zeros(n)
    drift[n_train:] = np.linspace(0.0, 2.0, n - n_train)  # starts after training
    y = X[:, 0] + drift + 0.1 * rng.normal(size=n)
    js = trained("drift", X[:n_train], y[:n_train], loss_function="RMSE", iterations=100, depth=4)
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    ote.fit(X[:n_train], y[:n_train], strategy="fixed", lam=1.0, max_iter=50)
    frozen = bw.ObliviousTreeEnsemble.from_bytes(ote.to_bytes())

    sse_frozen = sse_updated = 0.0
    for k in range(n_win):
        a, b = n_train + k * win, n_train + (k + 1) * win
        ote.update(X[a:b], y[a:b], steps=1)  # adapt before the next window
        c, d = b, min(b + win, n)
        if c < d:
            sse_frozen += float(np.sum((frozen.predict(X[c:d])[:, 0] - y[c:d]) ** 2))
            sse_updated += float(np.sum((ote.predict(X[c:d])[:, 0] - y[c:d]) ** 2))
    assert sse_updated < sse_frozen


def test_update_errors(trained):
    """Update before a fit and bad steps / shapes raise ValueError."""
    X = make_X(n=100, seed=31)
    y = make_y("Logloss", X, seed=31)
    js = trained("upd_err", X, y, loss_function="Logloss")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    with pytest.raises(ValueError):
        ote.update(X, y, steps=1)  # no fit before
    ote.fit(X, y, strategy="fixed", lam=1.0, max_iter=5)
    with pytest.raises(ValueError):
        ote.update(X, y, steps=0)
    with pytest.raises(ValueError):
        ote.update(X, y[:-1], steps=1)  # wrong y length


def test_classifier_update_label_check():
    """A label not in classes_ raises ValueError; known labels update fine."""
    X = make_X(n=120, seed=31)
    y = make_y("Logloss", X, seed=31)
    clf = bw.BoxwoodClassifier(catboost_params=dict(iterations=25, depth=3),
                              strategy="fixed", optimizer="tree_newton")
    clf.fit(X, y)
    clf.update(X, y, steps=1)
    bad = np.array(y, copy=True)
    bad[0] = 5.0
    with pytest.raises(ValueError):
        clf.update(X, bad, steps=1)
