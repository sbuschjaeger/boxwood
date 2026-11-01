# Python API

Generated from the docstrings of the `boxwood` package. Everything listed here is importable from `boxwood`
directly (`import boxwood as bw; bw.ObliviousTreeEnsemble`).

## `ObliviousTreeEnsemble`

::: boxwood.ensemble.ObliviousTreeEnsemble

## Estimators

::: boxwood.estimators.BoxwoodClassifier

::: boxwood.estimators.BoxwoodRegressor

## Helpers

::: boxwood.ensemble.category_strings

::: boxwood.ensemble.categorical_columns

## Module functions

`boxwood.backends() -> list[str]`
:   The names of the prediction backends this CPU can run, fastest first: some of `"avx512_vbmi"`, `"avx512"`,
    `"avx2"` (x86-64) or `"neon"` (64-bit ARM), and always `"scalar"`.

`boxwood.Error`
:   The exception type of the C++ core; it is raised as a `ValueError` subclass for invalid input, models and
    bytes.
