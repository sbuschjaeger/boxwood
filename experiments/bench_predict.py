"""Prediction speed of boxwood against CatBoost over several scenarios.

Usage: ``OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/bench_predict.py [scenario ...]`` (all if none given).
Needs the package built in build/release/python. Each scenario trains one CatBoost model (1,000 trees) on
synthetic data, imports it, and times ``predict_margin`` against CatBoost ``RawFormulaVal`` for several batch
sizes and thread counts (best of 5 runs; small batches are timed over many calls), with float32 and int8
leaves. Prints one markdown table row per batch size, thread count and leaf type: the times and the speedup
over CatBoost of every boxwood backend.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build" / "release" / "python"))
import boxwood as bw  # noqa: E402

N_ROWS = 50_000
N_FEATURES = 20
BATCHES = [1, 100, 10_000, N_ROWS]
THREADS = [1, 8]

# name: (depth, loss, number of categorical columns)
SCENARIOS = {
    "depth4": (4, "RMSE", 0),
    "depth6": (6, "RMSE", 0),
    "depth8": (8, "RMSE", 0),
    "multiclass3": (6, "MultiClass", 0),
    "categorical": (6, "RMSE", 4),
}


def make_data(n_cat, seed=0):
    """Synthetic rows: N_FEATURES numeric columns, then n_cat string columns with 20 values each."""
    rng = np.random.default_rng(seed)
    num = rng.normal(size=(N_ROWS, N_FEATURES)).astype(np.float32)
    y = num[:, 0] * 2.0 + num[:, 1] ** 2 + rng.normal(size=N_ROWS) * 0.1
    if n_cat == 0:
        return num, y
    X = pd.DataFrame(num, columns=[f"x{i}" for i in range(N_FEATURES)])
    for c in range(n_cat):
        codes = rng.integers(0, 20, size=N_ROWS)
        X[f"c{c}"] = [f"v{v}" for v in codes]
        y = y + (codes % 5 - 2) * 0.5
    return X, y


def best_time(f, calls):
    """Best of 5 runs of `calls` calls of f, per call, in seconds."""
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(calls):
            f()
        best = min(best, (time.perf_counter() - start) / calls)
    return best


def fmt(seconds):
    return f"{seconds * 1e3:.3f} ms" if seconds < 1e-2 else f"{seconds * 1e3:.1f} ms"


def run(name):
    depth, loss, n_cat = SCENARIOS[name]
    X, y = make_data(n_cat)
    if loss == "MultiClass":
        y = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))
    cat_cols = [f"c{c}" for c in range(n_cat)]
    model = CatBoost(dict(iterations=1000, depth=depth, loss_function=loss, thread_count=8,
                          random_seed=0, verbose=0, allow_writing_files=False))
    model.fit(X, y, cat_features=cat_cols or None)
    ote = bw.ObliviousTreeEnsemble.from_catboost(model)
    backends = bw.backends()
    print(f"\n### {name} (depth {depth}, {loss}, {n_cat} categorical columns, 1,000 trees)\n")
    print("| rows | threads | leaves | CatBoost | " + " | ".join(backends) + " |")
    print("|---|---|---|---|" + "---|" * len(backends))
    for n in BATCHES:
        Xb = X.iloc[:n] if isinstance(X, pd.DataFrame) else X[:n]
        # CatBoost needs a Pool to know the categorical columns; building it is part of its predict cost.
        cb_input = (lambda: Pool(Xb, cat_features=cat_cols)) if n_cat else (lambda: Xb)
        calls = max(1, 20_000 // n)
        for threads in THREADS:
            if n < 10_000 and threads > 1:
                continue  # too few rows to split
            cb = best_time(lambda: model.predict(cb_input(), prediction_type="RawFormulaVal",
                                                 thread_count=threads), calls)
            ote.set_threads(threads)
            for leaves in ("float32", "int8"):
                ote.set_leaf_type(leaves)
                cells = []
                for backend in backends:
                    ote.set_backend(backend)
                    t = best_time(lambda: ote.predict_margin(Xb), calls)
                    cells.append(f"{fmt(t)} ({cb / t:.2f}×)")
                print(f"| {n} | {threads} | {leaves} | {fmt(cb)} | " + " | ".join(cells) + " |", flush=True)
            # Back to the imported float32 leaves (int8 -> float32 keeps the rounded values).
            ote = bw.ObliviousTreeEnsemble.from_catboost(model)


if __name__ == "__main__":
    for scenario in sys.argv[1:] or SCENARIOS:
        run(scenario)
