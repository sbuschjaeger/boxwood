"""Leaf-refit tests: TreeNewton's objective against a scipy L-BFGS-B oracle.

The oracle rebuilds the same objective in numpy from ``ote.leaf_index(X)`` as a
sparse design matrix (one matrix row per output of every observation), with the anchor leaf
values read from the CatBoost JSON exactly as boxwood stores them (float32 of
``scale * leaf_value``; the bias is unscaled). Regression is fitted on ``y / sigma``.
"""

import json
import pickle

import numpy as np
import pytest
from scipy import optimize, sparse

import boxwood as bw
from test_ensemble import assert_close, make_X, make_y


def make_y3(X, seed=7):
    """A 3-class target (labels 0, 1, 2) from the same score as make_y."""
    score = X[:, 0] + 0.1 * X[:, 1] - 0.5 * np.log1p(X[:, 3])
    return np.digitize(score, [-0.3, 0.3])


def target(loss, X, seed):
    return make_y3(X, seed) if loss == "MultiClass" else make_y(loss, X, seed=seed)


def _sq(F, y):
    d = F - y
    return 0.5 * np.sum(d * d, axis=1), d


def _log(F, y):
    f = F[:, 0]
    sp = np.maximum(f, 0.0) + np.log1p(np.exp(-np.abs(f)))
    p = 1.0 / (1.0 + np.exp(-f))
    return sp - y[:, 0] * f, (p - y[:, 0])[:, None]


def _sm(F, y):
    m = F.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(F - m).sum(axis=1))
    p = np.exp(F - lse[:, None])
    return lse - np.sum(y * F, axis=1), p - y


LOSSFN = {"regression": _sq, "binary": _log, "multiclass": _sm}  # task -> (loss, grad) per observation


def _y_mat(y, task, K):
    """The target as a (n, K) matrix, matching what boxwood's fit sees."""
    a = np.asarray(y, dtype=np.float64)
    if task in ("regression", "binary"):
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    oh = np.zeros((a.size, K), dtype=np.float64)
    oh[np.arange(a.size), a.astype(int)] = 1.0
    return oh


def _sigma(task, y_mat, s):
    if task != "regression":
        return 1.0
    m = np.average(y_mat[:, 0], weights=s)
    v = np.average((y_mat[:, 0] - m) ** 2, weights=s)
    return float(np.sqrt(v)) if v > 0 else 1.0


def _anchor_bias(js, K):
    """Anchor leaf values (float32-rounded like boxwood) and the unscaled bias."""
    scale = js["scale_and_bias"][0]
    bias = np.asarray(js["scale_and_bias"][1], dtype=np.float64)
    leaf_begin, parts = [0], []
    for t in js["oblivious_trees"]:
        d = len(t.get("splits") or [])
        lv = (scale * np.asarray(t["leaf_values"], dtype=np.float64)).astype(np.float32).astype(np.float64)
        assert lv.size == (1 << d) * K
        parts.append(lv)
        leaf_begin.append(leaf_begin[-1] + lv.size)
    anchor = np.concatenate(parts) if parts else np.zeros(0)
    return anchor, bias, leaf_begin


def _design(leaf_index, leaf_begin, K, total):
    """Sparse (n*K) x (total+K) matrix: F = A @ [w; b]."""
    n, T = leaf_index.shape
    rows, cols = [], []
    ar = np.arange(n)
    for t in range(T):
        base = leaf_begin[t]
        for k in range(K):
            rows.append(ar * K + k)
            cols.append(base + leaf_index[:, t].astype(np.int64) * K + k)
    for k in range(K):
        rows.append(ar * K + k)
        cols.append(np.full(n, total + k, dtype=np.int64))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    return sparse.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n * K, total + K)).tocsr()


def _objective(A, n, K, total, y_scaled, s, lam, anchor_scaled, lossfn):
    S = s.sum()

    def f(theta):
        F = (A @ theta).reshape(n, K)
        loss, _ = lossfn(F, y_scaled)
        return np.average(loss, weights=s) + 0.5 * lam * np.sum((theta[:total] - anchor_scaled) ** 2)

    def grad(theta):
        F = (A @ theta).reshape(n, K)
        _, g = lossfn(F, y_scaled)
        gg = (A.T @ (g * s[:, None]).ravel()) / S
        gg[:total] += lam * (theta[:total] - anchor_scaled)
        return gg

    return f, grad


def _setup(js_text, X, y, task, s):
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
    K, T = ote.info()["n_outputs"], ote.info()["n_trees"]
    y_mat = _y_mat(y, task, K)
    sigma = _sigma(task, y_mat, s)
    anchor, bias, leaf_begin = _anchor_bias(json.loads(js_text), K)
    A = _design(ote.leaf_index(X), leaf_begin, K, leaf_begin[-1])
    return ote, K, T, sigma, anchor, bias, A, y_mat


@pytest.mark.parametrize("lam", [1e2, 1.0, 1e-2])
@pytest.mark.parametrize("loss,task", [("RMSE", "regression"), ("Logloss", "binary"),
                                      ("MultiClass", "multiclass")])
def test_matches_lbfgs_and_margins(trained, loss, task, lam):
    X = make_X(n=400, seed=7)
    y = target(loss, X, 7)
    js_text = trained(loss, X, y, loss_function=loss,
                     **({"classes_count": 3} if loss == "MultiClass" else {}))
    s = np.random.default_rng(3).uniform(0.5, 2.0, size=X.shape[0])
    ote, K, T, sigma, anchor, bias, A, y_mat = _setup(js_text, X, y, task, s)
    lossfn = LOSSFN[task]
    anchor_s, y_s = anchor / sigma, y_mat / sigma
    f, grad = _objective(A, X.shape[0], K, len(anchor), y_s, s, lam, anchor_s, lossfn)
    theta0 = np.concatenate([anchor_s, bias / sigma])
    res = optimize.minimize(f, theta0, jac=grad, method="L-BFGS-B",
                           options=dict(maxiter=20000, ftol=1e-15, gtol=1e-10))
    J_scipy = float(f(res.x))

    rep = ote.fit(X, y, sample_weight=s, strategy="fixed", lam=lam, max_iter=3000, tol=1e-13)
    assert abs(rep["objective_start"] - f(theta0)) <= 1e-9 * max(1.0, abs(f(theta0)))
    assert abs(rep["objective"] - J_scipy) <= 1e-6 * max(1.0, abs(J_scipy))

    # The refitted ensemble's margins match the scipy optimum (float32 leaves).
    # The softmax objective is invariant to a constant added to every output of a
    # observation and the bias is unpenalised, so the raw margins are only fixed up to that
    # constant; compare the centered margins for multiclass.
    margins = ote.predict_margin(X).reshape(-1, K)
    ref = (A @ res.x).reshape(-1, K) * sigma
    if K > 1:
        margins = margins - margins.mean(axis=1, keepdims=True)
        ref = ref - ref.mean(axis=1, keepdims=True)
    assert_close(margins, ref, tol=1e-4)


@pytest.mark.parametrize("loss,task", [("RMSE", "regression"), ("Logloss", "binary"),
                                      ("MultiClass", "multiclass")])
def test_objective_non_increasing(trained, loss, task):
    X = make_X(n=300, seed=11)
    y = target(loss, X, 11)
    js_text = trained(f"ni_{loss}", X, y, loss_function=loss,
                     **({"classes_count": 3} if loss == "MultiClass" else {}))
    prev = np.inf
    for it in (1, 2, 3, 5):
        ote = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
        rep = ote.fit(X, y, strategy="fixed", lam=1.0, max_iter=it)
        assert rep["objective"] <= prev + 1e-12
        prev = rep["objective"]


@pytest.mark.parametrize("loss,task", [("RMSE", "regression"), ("Logloss", "binary"),
                                      ("MultiClass", "multiclass")])
def test_holdout(trained, loss, task):
    X = make_X(n=300, seed=5)
    y = target(loss, X, 5)
    Xh = make_X(n=120, seed=99)
    yh = target(loss, Xh, 99)
    js_text = trained(f"ho_{loss}", X, y, loss_function=loss,
                     **({"classes_count": 3} if loss == "MultiClass" else {}))
    s = np.random.default_rng(4).uniform(0.5, 2.0, size=X.shape[0])
    sh = np.random.default_rng(8).uniform(0.0, 3.0, size=Xh.shape[0])
    ote, K, T, sigma, *_ = _setup(js_text, X, y, task, s)
    yh_mat = _y_mat(yh, task, K)

    rep = ote.fit(X, y, Xh, yh, sample_weight=s, holdout_weight=sh, lam=1.0, max_iter=20, patience=5)
    F = ote.predict_margin(Xh).reshape(-1, K) / sigma
    loss_np, _ = LOSSFN[task](F, yh_mat / sigma)
    assert_close(np.array([rep["holdout_loss"]]), np.array([np.average(loss_np, weights=sh)]), tol=1e-5)

    # With max_iter=1 the holdout loss is a finite number.
    ote1 = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
    rep1 = ote1.fit(X, y, Xh, yh, sample_weight=s, lam=1.0, max_iter=1)
    assert np.isfinite(rep1["holdout_loss"])


def test_cross_entropy_soft_targets(trained):
    X = make_X(n=400, seed=21)
    y = make_y("CrossEntropy", X, seed=21)  # soft probabilities in (0, 1)
    js_text = trained("ce", X, y, loss_function="CrossEntropy")
    s = np.random.default_rng(6).uniform(0.5, 2.0, size=X.shape[0])
    ote, K, T, sigma, anchor, bias, A, y_mat = _setup(js_text, X, y, "binary", s)
    anchor_s, y_s = anchor / sigma, y_mat / sigma
    f, grad = _objective(A, X.shape[0], K, len(anchor), y_s, s, 1.0, anchor_s, LOSSFN["binary"])
    theta0 = np.concatenate([anchor_s, bias / sigma])
    res = optimize.minimize(f, theta0, jac=grad, method="L-BFGS-B",
                           options=dict(maxiter=20000, ftol=1e-15, gtol=1e-10))
    rep = ote.fit(X, y, sample_weight=s, strategy="fixed", lam=1.0, max_iter=3000, tol=1e-13)
    assert abs(rep["objective"] - float(f(res.x))) <= 1e-6 * max(1.0, abs(float(f(res.x))))


def test_errors(trained):
    X = make_X(n=100, seed=31)
    y = make_y("Logloss", X, seed=31)
    js_bin = trained("err_bin", X, y, loss_function="Logloss")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js_bin)
    with pytest.raises(ValueError):
        ote.fit(X, y[:-1], strategy="fixed")  # wrong y length
    with pytest.raises(ValueError):
        ote.fit(X, y, strategy="fixed",
                sample_weight=np.where(np.arange(len(y)) == 0, -1.0, 1.0))  # negative weight

    js_mc = trained("err_mc", X, make_y3(X, 31), loss_function="MultiClass",
                    classes_count=3)
    ote_mc = bw.ObliviousTreeEnsemble.from_catboost_json(js_mc)
    bad = make_y3(X, 31).copy()
    bad[0] = 3  # label >= K
    with pytest.raises(ValueError):
        ote_mc.fit(X, bad, strategy="fixed")


@pytest.mark.parametrize("loss", ["Logloss", "MultiClass"])
def test_backtracking_keeps_objective_non_increasing(trained, loss):
    """Refit to targets that contradict the start margins (flipped labels).

    There the gradient is near 1 and the curvature near 0, so a plain Newton step
    overshoots; the step halving must keep the objective from increasing.
    """
    X = make_X(n=300, seed=13)
    y = target(loss, X, 13)
    js_text = trained(f"bt_{loss}", X, y, loss_function=loss, iterations=100, depth=4,
                      **({"classes_count": 3} if loss == "MultiClass" else {}))
    flipped = 1 - y if loss == "Logloss" else (y + 1) % 3
    prev = np.inf
    for it in (1, 2, 3):
        ote = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
        rep = ote.fit(X, flipped, strategy="fixed", lam=1e-4, max_iter=it)
        assert rep["objective"] <= min(prev, rep["objective_start"]) + 1e-12
        prev = rep["objective"]


# ---------------------------------------------------------------------------
# The Adam / GradientDescent optimizers, memory-mapped data.


def _mc(loss):
    return {"classes_count": 3} if loss == "MultiClass" else {}


TASKS = [("RMSE", "regression"), ("Logloss", "binary"), ("MultiClass", "multiclass")]


@pytest.mark.parametrize("optimizer", ["tree_newton", "adam"])
def test_memmap_matches_in_memory(trained, tmp_path, optimizer):
    """A fit on a numpy.memmap gives the bits of the fit on the array in memory."""
    X = make_X(n=300, seed=7)
    y = target("Logloss", X, 7)
    js_text = trained("Logloss", X, y, loss_function="Logloss")
    mm = np.memmap(tmp_path / "X.f32", dtype=np.float32, mode="w+", shape=X.shape)
    mm[:] = X
    mm.flush()
    mm = np.memmap(tmp_path / "X.f32", dtype=np.float32, mode="r", shape=X.shape)
    ote_mem = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
    ote_mm = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
    ote_mem.fit(X.astype(np.float32), y, strategy="fixed", optimizer=optimizer, max_iter=20)
    ote_mm.fit(mm, y, strategy="fixed", optimizer=optimizer, max_iter=20)
    assert ote_mem.to_bytes() == ote_mm.to_bytes()


@pytest.mark.parametrize("loss,task", TASKS)
def test_optimizers_reach_lbfgs(trained, loss, task):
    """Every full-batch optimizer reaches the scipy L-BFGS-B objective."""
    X = make_X(n=400, seed=7)
    y = target(loss, X, 7)
    js_text = trained(loss, X, y, loss_function=loss, **_mc(loss))
    s = np.random.default_rng(3).uniform(0.5, 2.0, size=X.shape[0])
    ote, K, T, sigma, anchor, bias, A, y_mat = _setup(js_text, X, y, task, s)
    anchor_s, y_s = anchor / sigma, y_mat / sigma
    f, grad = _objective(A, X.shape[0], K, len(anchor), y_s, s, 1.0, anchor_s, LOSSFN[task])
    theta0 = np.concatenate([anchor_s, bias / sigma])
    res = optimize.minimize(f, theta0, jac=grad, method="L-BFGS-B",
                           options=dict(maxiter=20000, ftol=1e-15, gtol=1e-10))
    J_scipy = float(f(res.x))
    scale = max(1.0, abs(J_scipy))

    ote_tn = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
    rep_tn = ote_tn.fit(X, y, sample_weight=s, strategy="fixed", lam=1.0, optimizer="tree_newton",
                        max_iter=3000, tol=1e-12)
    assert abs(rep_tn["objective"] - J_scipy) <= 1e-6 * scale, rep_tn

    ote_gd = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
    rep_gd = ote_gd.fit(X, y, sample_weight=s, strategy="fixed", lam=1.0,
                        optimizer="gradient_descent", max_iter=3000, tol=1e-12)
    assert abs(rep_gd["objective"] - J_scipy) <= 1e-6 * scale, rep_gd

    ote_ad = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
    rep_ad = ote_ad.fit(X, y, sample_weight=s, strategy="fixed", lam=1.0, optimizer="adam",
                        learning_rate=0.01, max_iter=3000, tol=1e-12)
    assert abs(rep_ad["objective"] - J_scipy) <= 1e-4 * scale, rep_ad


@pytest.mark.parametrize("loss,task", TASKS)
def test_non_increasing_gd(trained, loss, task):
    """Full-batch GradientDescent never raises the objective."""
    X = make_X(n=300, seed=11)
    y = target(loss, X, 11)
    js_text = trained(f"ni_{loss}", X, y, loss_function=loss, **_mc(loss))
    prev = np.inf
    for it in (1, 2, 3, 5):
        ote = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
        rep = ote.fit(X, y, strategy="fixed", lam=1.0, optimizer="gradient_descent", max_iter=it)
        assert rep["objective"] <= prev + 1e-12, (it, rep)
        prev = rep["objective"]


@pytest.mark.parametrize("loss", ["Logloss", "MultiClass"])
def test_backtracking_gd_non_increasing(trained, loss):
    """The same on the flipped-label case, where a plain gradient step overshoots."""
    X = make_X(n=300, seed=13)
    y = target(loss, X, 13)
    js_text = trained(f"bt_{loss}", X, y, loss_function=loss, iterations=100, depth=4, **_mc(loss))
    flipped = 1 - y if loss == "Logloss" else (y + 1) % 3
    prev = np.inf
    for it in (1, 2, 3):
        ote = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
        rep = ote.fit(X, flipped, strategy="fixed", lam=1e-4, optimizer="gradient_descent",
                      max_iter=it)
        assert rep["objective"] <= min(prev, rep["objective_start"]) + 1e-12, (it, rep)
        prev = rep["objective"]


@pytest.mark.parametrize("loss,task", TASKS)
def test_holdout_loss_reported(trained, loss, task):
    """The reported holdout loss matches the refitted ensemble's numpy holdout loss."""
    X = make_X(n=300, seed=5)
    y = target(loss, X, 5)
    Xh = make_X(n=120, seed=99)
    yh = target(loss, Xh, 99)
    js_text = trained(f"ho_{loss}", X, y, loss_function=loss, **_mc(loss))
    s = np.random.default_rng(4).uniform(0.5, 2.0, size=X.shape[0])
    ote, K, T, sigma, *_ = _setup(js_text, X, y, task, s)
    yh_mat = _y_mat(yh, task, K)
    rep = ote.fit(X, y, Xh, yh, sample_weight=s, strategy="early_stop", lam=1.0,
                  optimizer="tree_newton", max_iter=20, patience=5)
    F = ote.predict_margin(Xh).reshape(-1, K) / sigma
    loss_np, _ = LOSSFN[task](F, yh_mat / sigma)
    assert_close(np.array([rep["holdout_loss"]]), np.array([np.mean(loss_np)]), tol=1e-5)


def test_unknown_optimizer(trained):
    """An unknown optimizer name raises ValueError."""
    X = make_X(n=100, seed=31)
    y = make_y("Logloss", X, seed=31)
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(trained("err_bin", X, y, loss_function="Logloss"))
    with pytest.raises(ValueError):
        ote.fit(X, y, strategy="fixed", optimizer="sgd")


def test_sigma_with_a_large_target_mean(trained):
    """sigma = std(y) also when the mean of y is far larger than its spread."""
    X = make_X(n=400, seed=7)
    y = make_y("RMSE", X, seed=7)
    js_text = trained("RMSE", X, y, loss_function="RMSE")
    s = np.random.default_rng(3).uniform(0.5, 2.0, size=X.shape[0])
    for y_fit in (y, y + 1e8):
        ote, K, T, sigma, anchor, bias, A, y_mat = _setup(js_text, X, y_fit, "regression", s)
        f, _ = _objective(A, X.shape[0], K, len(anchor), y_mat / sigma, s, 1.0, anchor / sigma,
                          LOSSFN["regression"])
        theta0 = np.concatenate([anchor / sigma, bias / sigma])
        ote = bw.ObliviousTreeEnsemble.from_catboost_json(js_text)
        rep = ote.fit(X, y_fit, strategy="fixed", sample_weight=s, lam=1.0, max_iter=1)
        # J is about 1e15 for y + 1e8; a wrong sigma would change it by order 1.
        assert abs(rep["objective_start"] - f(theta0)) <= 1e-6 * max(1.0, abs(f(theta0)))


# ---------------------------------------------------------------------------
# Strategies for the strength lambda, the "no gain" rule and fit_lambda.


def _path_data():
    """A tiny binary problem: train on seed 5, holdout on seed 99."""
    X = make_X(n=300, seed=5)
    y = make_y("Logloss", X, seed=5)
    Xh = make_X(n=120, seed=99)
    yh = make_y("Logloss", Xh, seed=99)
    return X, y, Xh, yh


def _expected_path_count(holds):
    """Runs the path does for these holdout losses: stop after two without a
    new minimum over the runs."""
    best, bad = float("inf"), 0
    for i, h in enumerate(holds):
        if h < best:
            best, bad = h, 0
        else:
            bad += 1
            if bad == 2:
                return i + 1
    return len(holds)


def test_path_keeps_best_and_stops(trained):
    X, y, Xh, yh = _path_data()
    js = trained("m3c_path", X, y, loss_function="Logloss")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    rep = ote.fit(X, y, Xh, yh, strategy="path", max_iter=50)
    holds = [s["holdout_loss"] for s in rep["path"]]
    assert rep["holdout_loss"] == min([rep["holdout_loss_start"]] + holds)
    assert len(holds) == _expected_path_count(holds)


def test_verbose_prints_one_line_per_run(trained, capfd):
    X, y, Xh, yh = _path_data()
    js = trained("m3c_path", X, y, loss_function="Logloss")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    ote.fit(X, y, Xh, yh, strategy="path", max_iter=50)
    assert capfd.readouterr().err == ""
    rep = ote.fit(X, y, Xh, yh, strategy="path", max_iter=50, verbose=True)
    lines = capfd.readouterr().err.splitlines()
    assert len(lines) == len(rep["path"])
    assert all(line.startswith("boxwood fit: lambda ") for line in lines)


def test_path_warm_start_reaches_cold_optimum(trained):
    X, y, Xh, yh = _path_data()
    js = trained("m3c_path", X, y, loss_function="Logloss")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    rep = ote.fit(X, y, Xh, yh, strategy="path", max_iter=3000, tol=1e-12)
    step = next(s for s in rep["path"] if s["lambda"] == 1.0)
    ref = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    ref_rep = ref.fit(X, y, strategy="fixed", lam=1.0, max_iter=3000, tol=1e-12)
    assert abs(step["objective"] - ref_rep["objective"]) <= 1e-6 * max(
        1.0, abs(ref_rep["objective"]))


@pytest.mark.parametrize("strategy", ["path", "early_stop", "fixed"])
def test_no_gain_keeps_start(trained, strategy):
    """Fitting to random labels with the true labels as holdout changes nothing."""
    X = make_X(n=300, seed=13)
    y = make_y("Logloss", X, seed=13)
    js = trained("m3c_nogain", X, y, loss_function="Logloss")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    before = ote.predict_margin(X)
    rnd = np.random.default_rng(0).integers(0, 2, size=X.shape[0]).astype(np.float64)
    rep = ote.fit(X, rnd, X, y, strategy=strategy, max_iter=20)
    assert rep["no_gain"]
    assert np.array_equal(ote.predict_margin(X), before)


def test_early_stop(trained):
    X, y, Xh, yh = _path_data()
    js = trained("m3c_path", X, y, loss_function="Logloss")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    rep = ote.fit(X, y, Xh, yh, strategy="early_stop", lam=1e-4, max_iter=50)
    assert rep["holdout_loss"] <= rep["holdout_loss_start"]
    assert rep["lambda"] == 1e-4


def test_fit_lambda_info_and_pickle(trained):
    X, y, Xh, yh = _path_data()
    js = trained("m3c_path", X, y, loss_function="Logloss")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    assert np.isnan(ote.info()["fit_lambda"])
    ote.fit(X, y, Xh, yh, strategy="early_stop", lam=1e-4, max_iter=50)
    lam = ote.info()["fit_lambda"]
    assert lam == 1e-4
    data = ote.to_bytes()
    back = bw.ObliviousTreeEnsemble.from_bytes(data)
    assert back.info()["fit_lambda"] == lam
    assert back.to_bytes() == data
    again = pickle.loads(pickle.dumps(ote))
    assert again.info()["fit_lambda"] == lam
    assert again.to_bytes() == data


@pytest.mark.parametrize("optimizer", ["tree_newton", "adam", "gradient_descent"])
def test_path_every_optimizer(trained, optimizer):
    X, y, Xh, yh = _path_data()
    js = trained("m3c_path", X, y, loss_function="Logloss")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    rep = ote.fit(X, y, Xh, yh, strategy="path", optimizer=optimizer, max_iter=20)
    holds = [s["holdout_loss"] for s in rep["path"]]
    assert len(holds) >= 1
    assert rep["holdout_loss"] == min([rep["holdout_loss_start"]] + holds)


def test_strategy_errors(trained):
    X, y, _, _ = _path_data()
    js = trained("m3c_path", X, y, loss_function="Logloss")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    with pytest.raises(ValueError):
        ote.fit(X, y, strategy="path")  # no holdout
    with pytest.raises(ValueError):
        ote.fit(X, y, strategy="early_stop")  # no holdout
    with pytest.raises(ValueError):
        ote.fit(X, y, strategy="bogus")
