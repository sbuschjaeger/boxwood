"""The user-facing ensemble class around the native core ``boxwood._boxwood``."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from boxwood import _boxwood


def category_strings(values) -> list[str]:
    """Convert category values to the strings CatBoost and boxwood use.

    Args:
        values: An iterable of category values: strings, integers, floats or
            missing values.

    Returns:
        A list of strings: missing values (``None``, NaN, ``pandas.NA``)
        become ``"nan"``, everything else ``str(v)``, so the integer 3
        becomes "3".
    """
    # One numpy pass (str of every object) instead of a Python loop per value.
    arr = np.asarray(values, dtype=object)
    return np.where(pd.isna(arr), "nan", arr.astype(str)).tolist()


def _category_codes(values) -> "tuple[np.ndarray, list[str]]":
    """Dictionary-encode category values for the core.

    Args:
        values: An iterable of category values (see :func:`category_strings`).

    Returns:
        ``(codes, strings)``: a uint32 array with the index of every value in
        ``strings``, the distinct category strings. Each distinct value is
        converted (and hashed by the core) once.
    """
    arr = np.asarray(values, dtype=object)
    if pd.api.types.infer_dtype(arr, skipna=True).startswith("mixed"):
        # Values of different types can be equal (1 == True, 3 == 3.0) but
        # give different strings, so the strings are encoded instead.
        codes, uniques = pd.factorize(np.asarray(category_strings(arr)))
        return codes.astype(np.uint32), uniques.tolist()
    codes, uniques = pd.factorize(arr)  # missing values get the code -1
    strings = category_strings(uniques) + ["nan"]
    codes[codes < 0] = len(uniques)
    return codes.astype(np.uint32), strings


# Below this many observations the categorical columns go to the core as they are: factorizing costs a fixed time
# per column that a small batch does not win back by hashing fewer distinct values.
_FACTORIZE_FROM = 256


def _identity_codes(block: np.ndarray) -> "list[tuple[np.ndarray, list[str]]]":
    """Encode every column of a 2-D object array as its own dictionary, with the codes 0..n-1.

    Args:
        block: The categorical columns, shape ``(n, n_categorical)``.

    Returns:
        One ``(codes, strings)`` pair per column, as :func:`_category_codes` gives them; the values of the whole
        block are converted to strings at once.
    """
    codes = np.arange(len(block), dtype=np.uint32)
    return [(codes, column) for column in category_strings(block.T)]  # a 2-D input gives one list per column


def categorical_columns(X: pd.DataFrame) -> list:
    """The columns of a frame that CatBoost and boxwood treat as categorical.

    Args:
        X: A pandas DataFrame.

    Returns:
        The labels of the columns that are not numeric, plus the bool
        columns (a boolean is a category for a tree model, not a number).
    """
    return [c for c in X.columns
            if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]


def _as_weights(w) -> np.ndarray:
    """Per-observation weights as the C-contiguous float64 array the core reads (empty = all 1)."""
    return np.empty(0, dtype=np.float64) if w is None else np.ascontiguousarray(w, dtype=np.float64)


class ObliviousTreeEnsemble:
    """An oblivious tree ensemble imported from CatBoost.

    In an oblivious tree all nodes of one level share the same split, so a tree of
    depth ``d`` (0 to 8) has ``2**d`` leaves and an observation's leaf index has one bit per
    level. Build an ensemble with :meth:`from_catboost`, :meth:`from_catboost_json`
    or :meth:`from_bytes`; it is not meant to be constructed directly.
    """

    def __init__(self, inner: "_boxwood.Ensemble") -> None:
        """Wrap a native ensemble; prefer the ``from_*`` class methods.

        Args:
            inner: The native ensemble to wrap.
        """
        self._set_inner(inner)

    def _set_inner(self, inner: "_boxwood.Ensemble") -> None:
        """Wrap ``inner`` and cache the column indices every prediction needs.

        Args:
            inner: The native ensemble to wrap.
        """
        self._inner = inner
        info = inner.info()
        # The columns never change after the import, so predict does not ask
        # the core for them on every call.
        self._num_idx, self._cat_idx = info["numeric_flat_index"], info["categorical_flat_index"]

    @classmethod
    def from_catboost(cls, model, leaf_type: str = "float32") -> "ObliviousTreeEnsemble":
        """Import a trained CatBoost model.

        Args:
            model: A trained ``catboost.CatBoost`` model with oblivious trees of
                depth at most 8. Numeric, one-hot and CTR features (including
                combinations) are supported.
            leaf_type: ``"float32"`` (CatBoost's leaves rounded to float32) or
                ``"int8"`` (quantized, see :meth:`set_leaf_type`).

        Returns:
            The imported ensemble.

        Raises:
            ValueError: If the model has a tree deeper than 8, a loss function
                boxwood does not know, more than 255 borders or one-hot values
                per feature, or an unknown ``leaf_type``.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save_model(str(path), format="json")
            text = path.read_text(encoding="utf-8")
        return cls.from_catboost_json(text, leaf_type)

    @classmethod
    def from_catboost_json(cls, text: str, leaf_type: str = "float32") -> "ObliviousTreeEnsemble":
        """Import a CatBoost model given as JSON text.

        Args:
            text: The JSON written by ``CatBoost.save_model(format="json")``.
            leaf_type: ``"float32"`` or ``"int8"``, as in :meth:`from_catboost`.

        Returns:
            The imported ensemble.

        Raises:
            ValueError: If the JSON is malformed, a tree is deeper than 8, the
                loss function is unknown, a feature has more than 255 borders
                or one-hot values, a split does not match the feature data, or
                ``leaf_type`` is unknown.
        """
        ens = cls(_boxwood.Ensemble.from_catboost_json(text))
        if leaf_type != "float32":
            ens.set_leaf_type(leaf_type)
        return ens

    def _split_X(self, X) -> "tuple[np.ndarray, list[tuple[np.ndarray, list[str]]]]":
        """Split ``X`` into the numeric matrix and the categorical string columns.

        Args:
            X: A pandas DataFrame or a 2-D numpy array (numeric or object
                dtype) with all columns of the model, in the model's column
                order; ``info()["numeric_flat_index"]`` and
                ``info()["categorical_flat_index"]`` say which column is which.

        Returns:
            ``(numeric, categorical)``: a C-contiguous float32 array of shape
            ``(n, n_numeric)`` and, in the model's categorical order, one
            dictionary-encoded column per categorical feature (see
            :func:`_category_codes`; :func:`_identity_codes` below
            ``_FACTORIZE_FROM`` observations).

        Raises:
            ValueError: If ``X`` is not two-dimensional or its column count
                differs from ``n_numeric + n_categorical``.
        """
        if not isinstance(X, pd.DataFrame):
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError(f"X must be 2-D, got {X.ndim} dimension(s)")
        num_idx, cat_idx = self._num_idx, self._cat_idx
        n_cols = len(num_idx) + len(cat_idx)
        if X.shape[1] != n_cols:
            raise ValueError(f"X has {X.shape[1]} columns, the model has {n_cols}")
        small = bool(cat_idx) and len(X) < _FACTORIZE_FROM
        if small and isinstance(X, pd.DataFrame):
            # One conversion of the whole frame costs less than selecting its numeric and its categorical columns;
            # na_value turns pandas' missing values (pd.NA of nullable columns) into NaN.
            X = X.to_numpy(dtype=object, na_value=np.nan)
        if isinstance(X, pd.DataFrame):
            # na_value: older pandas cannot convert pd.NA of nullable columns to float.
            numeric, columns = X.iloc[:, num_idx].to_numpy(dtype=np.float32, na_value=np.nan), lambda c: X.iloc[:, c]
        else:
            # An all-numeric float32 array in C order goes to the core without a copy.
            numeric, columns = (X if not cat_idx else X[:, num_idx]), lambda c: X[:, c]
        if small:
            cats = _identity_codes(np.asarray(columns(cat_idx), dtype=object))
        else:
            cats = [_category_codes(columns(c)) for c in cat_idx]
        return np.ascontiguousarray(numeric, dtype=np.float32), cats

    def predict(self, X) -> np.ndarray:
        """Predict the task output of every observation of ``X``.

        For a classifier this is the probability obtained from the margins
        (sigmoid for binary, softmax for multiclass); for regression it is the
        margin itself.

        Args:
            X: A pandas DataFrame or a 2-D numpy array (numeric or object
                dtype) with all columns of the model, in the model's column
                order; ``info()["numeric_flat_index"]`` and
                ``info()["categorical_flat_index"]`` say which column is which.
                Numeric missing values are NaN; categorical values are
                converted with :func:`category_strings`.

        Returns:
            A float64 array of shape ``(n, K)`` with ``K = info()["n_outputs"]``.

        Raises:
            ValueError: If ``X`` is not 2-D or its column count differs from
                ``n_numeric + n_categorical``.
        """
        numeric, cats = self._split_X(X)
        return self._inner.predict(numeric, cats)

    def predict_margin(self, X) -> np.ndarray:
        """Predict the raw margin (``RawFormulaVal``) of every observation of ``X``.

        The margin is the bias plus the sum of the leaf values over all trees,
        before the task link.

        Args:
            X: A pandas DataFrame or a 2-D numpy array (numeric or object
                dtype) with all columns of the model, in the model's column
                order (see :meth:`predict`).

        Returns:
            A float64 array of shape ``(n, K)``.

        Raises:
            ValueError: If ``X`` is not 2-D or its column count differs from
                ``n_numeric + n_categorical``.
        """
        numeric, cats = self._split_X(X)
        return self._inner.predict_margin(numeric, cats)

    def leaf_index(self, X) -> np.ndarray:
        """The leaf each observation of ``X`` lands in, for every tree.

        Args:
            X: A pandas DataFrame or a 2-D numpy array (numeric or object
                dtype) with all columns of the model, in the model's column
                order (see :meth:`predict`).

        Returns:
            A uint8 array of shape ``(n, n_trees)``; entry ``(i, t)`` is the
            leaf index of observation ``i`` in tree ``t``.

        Raises:
            ValueError: If ``X`` is not 2-D or its column count differs from
                ``n_numeric + n_categorical``.
        """
        numeric, cats = self._split_X(X)
        return self._inner.leaf_index(numeric, cats)

    def _prepare_y(self, y) -> np.ndarray:
        """Convert ``y`` to the flat float64 layout the native fit expects.

        Args:
            y: The target. Regression and binary: a 1-D array of ``n`` values.
                Multiclass: integer labels ``0..K-1`` (one-hot encoded) or a
                2-D ``(n, K)`` array of probabilities.

        Returns:
            A flat C-contiguous float64 array of ``n * K`` values.

        Raises:
            ValueError: If a regression / binary target is not 1-D, or a
                multiclass label is outside ``0..K-1``.
        """
        info = self.info()
        K = info["n_outputs"]
        arr = np.asarray(y)
        if info["task"] in ("regression", "binary"):
            out = np.ascontiguousarray(arr, dtype=np.float64)
            if out.ndim != 1:
                raise ValueError(f"{info['task']} y must be 1-D, got {out.ndim} dimension(s)")
            return out
        if arr.ndim == 2:
            return np.ascontiguousarray(arr, dtype=np.float64).ravel()  # probabilities
        labels = arr.astype(np.int64)
        if labels.size and (labels.min() < 0 or labels.max() >= K):
            raise ValueError(f"multiclass labels must be in 0..{K - 1}")
        onehot = np.zeros((labels.size, K), dtype=np.float64)
        onehot[np.arange(labels.size), labels] = 1.0
        return onehot.ravel()

    def fit(
        self,
        X,
        y,
        X_holdout=None,
        y_holdout=None,
        *,
        strategy: str = "path",
        lam: float | None = None,
        optimizer: str = "tree_newton",
        learning_rate: float = 0.0,
        sample_weight=None,
        holdout_weight=None,
        max_iter: int = 200,
        tol: float = 1e-3,
        patience: int = 10,
        split_budget: int | None = None,
        max_bytes: int | None = None,
    ) -> dict:
        """Refit the leaf values and the bias, keeping the splits fixed.

        Runs ``J(w, b) = mean weighted loss + (lam / 2) * ||w - a||^2`` where
        ``a`` is the current leaf values (the anchor) and the bias is not
        penalised. Everything is computed in float64 (regression on ``y /
        std(y)``); the result is written back as float32 leaves and a float64
        bias. The features are read once (with a budget twice), to compute
        every observation's leaf index in every tree; the fit then holds these
        ``n * n_trees`` bytes and the targets and weights. So ``X`` can be a ``numpy.memmap`` of data larger
        than memory (an all-numeric float32 C-order array is read in place,
        without a copy). For data that arrives over time, or does not fit even
        so, use :meth:`update` window by window.

        The strategy chooses the strength: ``"fixed"`` runs once with ``lam``;
        ``"path"`` runs the warm-started path ``lam = 1e2, 1e1, ..., 1e-4``,
        keeps the best run on the holdout and stops after two lambdas without
        improvement; ``"early_stop"`` runs once with a small ``lam`` and keeps
        the best iterate by holdout loss. With a holdout, a run that is not
        strictly better than the start keeps the start parameters untouched
        (``no_gain``). The lambda of the best run is stored in the ensemble and
        reported as ``fit_lambda`` in :meth:`info`.

        With ``split_budget = B`` the ensemble is first pruned to at most ``B``
        split evaluations per observation (the sum of the kept tree depths): each tree
        is written in the level basis, iterative hard thresholding picks the
        kept levels under the fit objective, the trees are cut to them and the
        leaf values are then refit as above. ``max_bytes = M`` prunes the same
        way to at most ``M`` bytes of :meth:`to_bytes` in the current leaf
        type (a level then costs its bytes instead of one split). A budget is
        a hard limit: the pruned ensemble is kept even when the holdout loss
        is worse than without pruning.

        Args:
            X: The training features (see :meth:`predict`).
            y: The training target (see :meth:`_prepare_y` for the per-task
                shape).
            X_holdout: Optional holdout features; required by ``"path"`` and
                ``"early_stop"``.
            y_holdout: The holdout target, same shape rules as ``y``.
            strategy: ``"path"`` (default), ``"early_stop"`` or ``"fixed"``.
            lam: The penalty strength; ``None`` means 1.0 for ``"fixed"`` and
                1e-4 for ``"early_stop"``; ``"path"`` ignores it.
            optimizer: ``"tree_newton"``, ``"adam"`` or ``"gradient_descent"``.
            learning_rate: The step size; 0 uses the optimizer's default.
            sample_weight: Optional per-observation weights (``n`` values,
                non-negative).
            holdout_weight: Optional per-observation weights of the holdout.
            max_iter: Maximum number of iterations per run.
            tol: Relative objective drop that counts as converged.
            patience: Holdout iterations without improvement before stopping.
            split_budget: Optional cap on the split evaluations per observation after
                pruning; ``None`` (default) does not prune.
            max_bytes: Optional cap on the size of :meth:`to_bytes` after
                pruning, without the state of online updates with Adam;
                ``None`` (default) does not prune. Not together with
                ``split_budget``.

        Returns:
            A dict with ``objective_start`` (J at the anchor), ``objective``
            (J at the returned parameters), ``holdout_loss`` (NaN without a
            holdout), ``iterations``, ``converged``, ``lambda`` (the strength
            of the returned parameters, NaN with no gain), ``no_gain``,
            ``holdout_loss_start`` (holdout loss of the start parameters, NaN
            without a holdout), ``path`` (one dict per run with ``lambda``,
            ``objective``, ``holdout_loss`` and ``iterations``),
            ``splits_used`` (the sum of the tree depths after the fit) and
            ``trees_removed`` (the number of trees the cut removed).

        Raises:
            ValueError: If the strategy or optimizer name is unknown,
                ``"path"`` or ``"early_stop"`` is used without a holdout, a
                target or weight has the wrong length, a weight is negative,
                ``lam`` or ``learning_rate`` is negative, a holdout is
                requested without observations, both budgets are given, or
                ``max_bytes`` is smaller than the model without trees.
        """
        if lam is None:
            lam = 1.0 if strategy == "fixed" else 1e-4
        if X_holdout is not None:
            hold_num, hold_cats = self._split_X(X_holdout)
            hold_y = self._prepare_y(y_holdout)
            hold_w = _as_weights(holdout_weight)
        else:
            hold_num = np.empty((0, 0), dtype=np.float32)
            hold_cats = [_category_codes([]) for _ in range(self.info()["n_categorical"])]
            hold_y = hold_w = _as_weights(None)
        numeric, cats = self._split_X(X)
        y_arr = self._prepare_y(y)
        w_arr = _as_weights(sample_weight)
        return self._inner.fit(
            numeric, cats, y_arr, w_arr, hold_num, hold_cats, hold_y, hold_w,
            lam, optimizer, learning_rate, max_iter, tol, patience, strategy, split_budget, max_bytes,
        )

    def update(self, X, y, steps: int = 1, sample_weight=None) -> dict:
        """Adapt the leaf values and the bias to a window of new data.

        The call for data that arrives over time, or that is too large for
        one :meth:`fit`: one call per window, for example per chunk of a
        stream or per slice of a memory-mapped array. Runs ``steps``
        iterations of the optimizer of the last
        :meth:`fit` (with its learning rate) on the objective of that fit
        over the window: the anchor is the current leaf values, ``lam`` and
        ``sigma`` are the stored ``fit_lambda`` and ``fit_sigma`` -- never
        recomputed from the window, so the units stay those of the fit. Adam
        continues from the state stored in the ensemble and stores its new
        state there. Splits, borders and CTR tables stay fixed.

        Args:
            X: The window features (see :meth:`predict`).
            y: The window target, same shape rules as in :meth:`fit`.
            steps: Number of optimizer iterations; must be >= 1.
            sample_weight: Optional per-observation weights (``n`` values,
                non-negative).

        Returns:
            A dict with ``objective_start`` (the window objective at the
            leaf values before the call), ``objective`` (after the steps)
            and ``steps``.

        Raises:
            ValueError: If no :meth:`fit` happened before, ``steps`` < 1,
                or a target or weight has the wrong length.
        """
        numeric, cats = self._split_X(X)
        y_arr = self._prepare_y(y)
        w_arr = _as_weights(sample_weight)
        return self._inner.update(numeric, cats, y_arr, w_arr, steps)

    def splits(self) -> "list[list[tuple[str, int, str, int]]]":
        """The splits of every tree, for inspection.

        Returns:
            One list per tree, ordered from the root level to the leaf level.
            Each split is a tuple ``(source, index, kind, arg)``: ``source`` is
            ``"numeric"`` (``index`` is the column of ``X``), ``"onehot"``
            (``index`` is the categorical feature index) or ``"ctr"`` (``index``
            is the CTR feature), ``kind`` is ``"gt"`` (``code > arg``) or
            ``"eq"`` (``code == arg``), and ``arg`` is the index of the border
            or value inside that feature. A depth-0 tree has an empty list.
        """
        n_trees = self._inner.info()["n_trees"]
        return [self._inner.splits(t) for t in range(n_trees)]

    def leaves(self) -> "list[np.ndarray]":
        """The leaf values of every tree, for inspection.

        Returns:
            One float64 array of shape ``(2**depth, n_outputs)`` per tree; row
            ``j`` is the leaf with index ``j`` (bit ``l`` of ``j`` is the outcome
            of the split at level ``l``). Int8 leaves are dequantized.
        """
        info = self._inner.info()
        k = info["n_outputs"]
        return [np.asarray(self._inner.leaves(t)).reshape(-1, k) for t in range(info["n_trees"])]

    def bias(self) -> np.ndarray:
        """The bias, one float64 value per output.

        Returns:
            An array of shape ``(n_outputs,)``.
        """
        return np.asarray(self._inner.bias())

    def set_backend(self, name: str) -> None:
        """Choose the prediction backend.

        Args:
            name: ``"scalar"``, ``"avx2"``, ``"avx512"``, ``"avx512_vbmi"``
                or ``"neon"``; ``""`` selects the best backend
                available on this CPU (see :func:`boxwood.backends`). The
                choice is not part of the serialized model.

        Raises:
            ValueError: If the name is unknown or the backend is not
                available on this CPU.
        """
        self._inner.set_backend(name)

    def set_threads(self, n: int) -> None:
        """Choose how many threads ``predict``, ``predict_margin`` and
        ``leaf_index`` use.

        Each thread takes a range of 256-observation blocks; inputs of fewer than
        1,024 observations per thread use fewer threads. The result does not depend
        on the number of threads. The choice is not part of the serialized
        model.

        Args:
            n: The number of threads; ``0`` uses one per hardware thread.
                The default is 1.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        self._inner.set_threads(n)

    def set_leaf_type(self, name: str) -> None:
        """Change how the leaf values are stored.

        ``"int8"`` quantizes the float32 leaves in blocks of 64 consecutive
        trees: with ``M`` the largest absolute leaf value of a block, a tree
        gets the step ``(M / 127) / 2**d`` with the largest ``d <= 7`` for
        which its own largest absolute leaf value is at most ``M / 2**d``,
        and int8 codes ``q = round-half-away-from-zero(value / step)``
        clamped to ``[-127, 127]``; the value of a leaf is ``q * step``.
        Prediction sums the codes of a block as integers (exact), which makes
        int8 faster than float32; a margin moves by at most half the sum of
        the steps. ``"float32"`` stores the dequantized values again (rounded
        to float32). The leaf type is part of the serialized model and shows
        up in ``info()["leaf_type"]``.

        Args:
            name: ``"int8"`` or ``"float32"``.

        Raises:
            ValueError: If the name is neither ``"int8"`` nor ``"float32"``.
        """
        self._inner.set_leaf_type(name)

    def info(self) -> dict:
        """The shape of the ensemble.

        Returns:
            A dict with ``task`` (``"regression"``, ``"binary"`` or
            ``"multiclass"``), ``n_outputs`` (K), ``n_numeric`` (the number of
            numeric columns ``X`` must have), ``n_categorical`` (the number of
            categorical columns), ``n_trees``, ``n_splits``, ``tree_bytes``
            (the bytes of :meth:`to_bytes` the trees take; the rest is the
            input encoding, the bias and the header), ``leaf_type``
            (``"float32"`` or ``"int8"``),
            ``numeric_flat_index`` / ``categorical_flat_index`` (the CatBoost
            column of every numeric / categorical feature), ``fit_lambda``
            (the strength of the last :meth:`fit`, NaN before the first one),
            ``fit_sigma`` (the scale of the last fit, 1 for classification,
            NaN before the first one) and ``fit_optimizer`` (the name of the
            optimizer of the last fit, ``""`` before the first one).
        """
        return self._inner.info()

    def to_bytes(self) -> bytes:
        """Serialize the ensemble.

        Floats are stored raw, so a round trip through :meth:`from_bytes` is
        bit-exact.

        Returns:
            The serialized model.
        """
        return self._inner.to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ObliviousTreeEnsemble":
        """Load a model saved by :meth:`to_bytes`.

        Args:
            data: The bytes produced by :meth:`to_bytes`.

        Returns:
            The loaded ensemble.

        Raises:
            ValueError: If the bytes are damaged, truncated, of another
                version, or internally inconsistent.
        """
        return cls(_boxwood.Ensemble.from_bytes(data))

    def __getstate__(self) -> bytes:
        """The serialized model, so ``pickle`` and ``joblib`` work.

        Returns:
            The same bytes as :meth:`to_bytes`.
        """
        return self.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        """Rebuild the ensemble from :meth:`__getstate__` output.

        Args:
            state: The bytes produced by :meth:`__getstate__`.

        Raises:
            ValueError: If the bytes are damaged or of another version.
        """
        self._set_inner(_boxwood.Ensemble.from_bytes(state))
