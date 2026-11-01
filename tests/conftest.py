"""Make the tests run against the package built in ``build/release/python`` (or the
directory in ``BOXWOOD_PKG_DIR``, e.g. an unpacked wheel), and train the tiny CatBoost
models the tests share."""

import os
import pathlib
import sys
import tempfile

import pytest

PKG_DIR = pathlib.Path(os.environ.get(
    "BOXWOOD_PKG_DIR", pathlib.Path(__file__).resolve().parent.parent / "build" / "release" / "python"))
if not (PKG_DIR / "boxwood").is_dir():
    raise RuntimeError(f"{PKG_DIR}/boxwood is missing; build it with: "
                       "cmake --preset release && cmake --build build/release -j8")
sys.path.insert(0, str(PKG_DIR))

from catboost import CatBoost  # noqa: E402

# Small and deterministic: the tests run next to other jobs on a shared machine.
TINY = dict(iterations=100, depth=4, thread_count=2, allow_writing_files=False, verbose=0,
            random_seed=0)


def train_catboost(X, y, cat_features=None, **params):
    """A trained tiny CatBoost model; `params` override the TINY defaults."""
    model = CatBoost({**TINY, **params})
    model.fit(X, y, cat_features=cat_features)
    return model


def catboost_json(model, pool=None) -> str:
    """The model's JSON export (with `pool`, it includes the category hash table)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "model.json"
        model.save_model(str(path), format="json", pool=pool)
        return path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def trained():
    """Train each named model once per session; returns get(name, X, y, **params) -> JSON text.

    The defaults are smaller than TINY (25 iterations, depth 3): the fit tests solve
    each objective to high precision with scipy.
    """
    cache = {}

    def get(name, X, y, **params):
        if name not in cache:
            cache[name] = catboost_json(train_catboost(X, y, **{"iterations": 25, "depth": 3, **params}))
        return cache[name]

    return get
