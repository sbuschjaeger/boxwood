"""Int8 leaf tests: the quantization error against the float32 margins,
fit and update with int8 leaves, the pickle round trip, and backend equality.

The margin bound: the trees form blocks of 64; with M = the largest
|leaf| of a block, tree t gets the step (M / 127) / 2^d with the largest
d <= 7 for which its own largest |leaf| is at most M / 2^d (read from the
model's float32 leaf values). A margin moves by at most ``sum_t step_t / 2``;
the int8 sums are exact, so only float64 rounding comes on top.
"""

import json
import pickle

import numpy as np
import pytest

import boxwood as bw
from test_ensemble import make_X, make_y


def _step_bound(js_text):
    """Half the sum of the int8 quantization steps of the trees of a model JSON."""
    j = json.loads(js_text)
    scale = np.float32(j["scale_and_bias"][0])
    tree_max = np.array([
        np.max(np.abs((scale * np.asarray(t["leaf_values"], dtype=np.float32)).astype(np.float32)))
        for t in j["oblivious_trees"]])
    total = 0.0
    for b in range(0, len(tree_max), 64):
        block = tree_max[b:b + 64]
        M = block.max()
        for m in block:
            d = 0
            while d < 7 and m <= M / 2.0 ** (d + 1):
                d += 1
            total += M / 127.0 / 2.0 ** d
    return total / 2.0


def test_int8_margins_within_half_a_step(trained):
    X = make_X(n=500, seed=11)
    y = make_y("RMSE", X, seed=11)
    js = trained("int8_rmse", X, y, loss_function="RMSE")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    ote.set_backend("scalar")
    m32 = ote.predict_margin(X)
    bound = _step_bound(js)
    assert bound > 0

    ote.set_leaf_type("int8")
    assert ote.info()["leaf_type"] == "int8"
    m8 = ote.predict_margin(X)
    slack = 1e-9 * max(1.0, float(np.abs(m32).max()))
    assert float(np.max(np.abs(m8 - m32))) <= bound + slack


def test_import_as_int8_equals_set_leaf_type(trained):
    X = make_X(n=300, seed=12)
    y = make_y("RMSE", X, seed=12)
    js = trained("int8_rmse", X, y, loss_function="RMSE")
    direct = bw.ObliviousTreeEnsemble.from_catboost_json(js, leaf_type="int8")
    later = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    later.set_leaf_type("int8")
    assert direct.info()["leaf_type"] == "int8"
    assert direct.to_bytes() == later.to_bytes()


def test_fit_and_update_keep_int8(trained):
    X = make_X(n=400, seed=13)
    y = make_y("RMSE", X, seed=13)
    Xh = make_X(n=150, seed=14)
    yh = make_y("RMSE", Xh, seed=14)
    js = trained("int8_fit", X, y, loss_function="RMSE")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    ote.set_leaf_type("int8")

    rep = ote.fit(X, y, Xh, yh, strategy="path", max_iter=30)
    assert rep["objective"] <= rep["objective_start"] + 1e-9
    assert ote.info()["leaf_type"] == "int8"  # the result is quantized

    upd = ote.update(X, y, steps=5)
    assert upd["steps"] == 5
    assert ote.info()["leaf_type"] == "int8"


def test_pickle_round_trip_int8(trained):
    X = make_X(n=200, seed=15)
    y = make_y("RMSE", X, seed=15)
    js = trained("int8_pickle", X, y, loss_function="RMSE")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    ote.set_leaf_type("int8")
    back = pickle.loads(pickle.dumps(ote))
    assert back.info()["leaf_type"] == "int8"
    assert back.to_bytes() == ote.to_bytes()
    assert np.array_equal(back.predict_margin(X), ote.predict_margin(X))


def test_backends_match_with_int8(trained):
    X = make_X(n=600, seed=17)
    y = make_y("RMSE", X, seed=17)
    js = trained("int8_be", X, y, loss_function="RMSE")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    ote.set_leaf_type("int8")
    ote.set_backend("scalar")
    base = ote.predict_margin(X)
    for name in bw.backends():
        ote.set_backend(name)
        assert np.array_equal(ote.predict_margin(X), base)  # bit-exact


def test_leaf_type_round_trip_and_errors(trained):
    X = make_X(n=200, seed=19)
    y = make_y("RMSE", X, seed=19)
    js = trained("int8_rt", X, y, loss_function="RMSE")
    ote = bw.ObliviousTreeEnsemble.from_catboost_json(js)
    assert ote.info()["leaf_type"] == "float32"

    ote.set_leaf_type("int8")
    m8 = ote.predict_margin(X)
    ote.set_leaf_type("float32")
    assert ote.info()["leaf_type"] == "float32"
    # The dequantized values are kept (rounded to float32), so int8 again
    # gives the same codes and steps.
    assert np.allclose(ote.predict_margin(X), m8, rtol=1e-6, atol=1e-9)
    ote.set_leaf_type("int8")
    assert np.array_equal(ote.predict_margin(X), m8)

    with pytest.raises(ValueError):
        ote.set_leaf_type("int16")
