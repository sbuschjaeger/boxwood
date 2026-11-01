# Benchmark against CatBoost

`experiments/bench_predict.py` measures `predict_margin` against CatBoost's `predict(prediction_type="RawFormulaVal")`
on the same model, for several scenarios:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/bench_predict.py              # all scenarios
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/bench_predict.py depth6 categorical
```

| scenario | model |
|---|---|
| `depth4`, `depth6`, `depth8` | regression, 1,000 trees of that depth, 20 numeric features |
| `multiclass3` | 3 classes, depth 6 |
| `categorical` | regression, depth 6, 20 numeric and 4 categorical columns (CatBoost builds several hundred CTR features from them) |

Each scenario trains its CatBoost model on 50,000 synthetic observations (8 threads) and then times batches of 1, 100,
10,000 and 50,000 observations with 1 and 8 threads, with float32 and with int8 leaves, for every backend of the CPU.
A time is the best of 5 runs; small batches are timed over many calls. The output is one markdown table per
scenario, with the speedup over CatBoost in parentheses.

For a fair comparison both libraries get the same number of threads, and CatBoost's time for a categorical
model includes building its `Pool` (it needs one to know the categorical columns), as boxwood's includes the
conversion of the string columns.

The results on one machine are in [Performance](../reference/performance.md).

!!! tip
    Close other work while benchmarking: the times are best-of-5 wall-clock times, and a busy machine makes
    both libraries slower by different amounts.
