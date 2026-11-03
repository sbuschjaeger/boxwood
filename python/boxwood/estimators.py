"""sklearn-style estimators: train CatBoost, import it, refit the leaves with boxwood."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split

from boxwood.ensemble import ObliviousTreeEnsemble, categorical_columns, category_strings

_CATBOOST_DEFAULTS = dict(
    learning_rate=0.05,
    depth=6,
    iterations=10000,
    use_best_model=True,
    allow_writing_files=False,
)


def early_stopping_rounds(n: int) -> int:
    """AutoGluon's early-stopping patience for CatBoost on ``n`` observations.

    300 rounds up to 10,000 observations, then ``max(20, round(300 * 10000 / n))``:
    on large data a round gains less, and each costs more.

    Args:
        n: The number of observations CatBoost trains on.

    Returns:
        The number of rounds without holdout improvement before stopping.
    """
    return 300 if n <= 10000 else max(20, round(300 * 10000 / n))


def _to_frame(X) -> pd.DataFrame:
    """``X`` as a pandas DataFrame; a 2-D numpy array is converted.

    The dtypes of a converted array are inferred so an object array holding
    floats and strings reads like the equivalent frame.

    Args:
        X: A pandas DataFrame or a 2-D numpy array.

    Returns:
        The frame.

    Raises:
        ValueError: If a numpy input is not two-dimensional.
    """
    if isinstance(X, pd.DataFrame):
        return X
    A = np.asarray(X)
    if A.ndim != 2:
        raise ValueError(f"X must be 2-D, got {A.ndim} dimension(s)")
    return pd.DataFrame(A).infer_objects()


def _stringified(X: pd.DataFrame, cat_pos) -> pd.DataFrame:
    """A copy of ``X`` with the categorical columns replaced by their strings."""
    out = X.copy()
    for p in cat_pos:
        out.iloc[:, p] = category_strings(out.iloc[:, p])
    return out


class _BoxwoodBase(BaseEstimator):
    """Shared fit of the boxwood estimators: holdout, CatBoost, import, refit.

    Subclasses set ``_classifier`` and implement ``_encode(y)``, which
    returns ``(loss, target)``; the public ``fit`` lives in the subclasses so
    its signature stays explicit.
    """

    _classifier = False

    def __init__(self, catboost_params=None, strategy="path",
                 optimizer="tree_newton", split_budget=None, max_bytes=None,
                 leaf_type="int8", holdout_fraction=0.125, random_state=0, verbose=False):
        """Store the parameters; nothing happens until :meth:`fit`.

        Args:
            catboost_params: Extra parameters for the CatBoost constructor,
                applied on top of ``learning_rate=0.05, depth=6,
                iterations=10000, use_best_model=True,
                random_seed=random_state, allow_writing_files=False``,
                ``early_stopping_rounds`` from :func:`early_stopping_rounds`
                (AutoGluon's rule) and ``verbose`` (100 with ``verbose``, else 0). ``loss_function`` is set by the target
                and cannot be overridden.
            strategy: The leaf-refit strength strategy: ``"path"``,
                ``"early_stop"`` or ``"fixed"`` (see
                :meth:`ObliviousTreeEnsemble.fit`).
            optimizer: The leaf-refit optimizer: ``"tree_newton"``,
                ``"adam"`` or ``"gradient_descent"``.
            split_budget: Optional cap on the split evaluations per observation
                after pruning (see ``split_budget`` in
                :meth:`ObliviousTreeEnsemble.fit`); ``None`` does not prune.
            max_bytes: Optional cap on the size of the fitted ensemble's
                :meth:`ObliviousTreeEnsemble.to_bytes` (see ``max_bytes`` in
                :meth:`ObliviousTreeEnsemble.fit`); ``None`` does not prune.
                Not together with ``split_budget``.
            leaf_type: The leaf type of the fitted ensemble: ``"int8"``
                (a quarter of the leaf memory) or ``"float32"`` (see
                :meth:`ObliviousTreeEnsemble.set_leaf_type`). The trees are
                imported in this type, so ``max_bytes`` counts it.
            holdout_fraction: The fraction of the training observations held off
                for early stopping and the leaf refit when no holdout is
                given to :meth:`fit`.
            random_state: The seed for the holdout split and the CatBoost
                training.
            verbose: Print the progress: CatBoost's holdout metric every 100
                iterations and one line per run of the leaf refit.
        """
        self.catboost_params = catboost_params
        self.strategy = strategy
        self.optimizer = optimizer
        self.split_budget = split_budget
        self.max_bytes = max_bytes
        self.leaf_type = leaf_type
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state
        self.verbose = verbose

    def _fit_common(self, X, y, X_holdout, y_holdout, sample_weight):
        """Run the shared pipeline and return ``self``.

        Splits off the holdout when none is given, trains CatBoost on the
        training observations, imports the trees as ``leaf_type`` and refits
        the leaf values on them (pruned first with a budget).

        Args:
            X: Training features, a DataFrame or 2-D numpy array.
            y: Training target, encoded by the subclass ``_encode``.
            X_holdout: Optional holdout features; ``holdout_fraction`` of
                the training observations are split off ``X`` when it is None.
            y_holdout: The holdout target; required with ``X_holdout``.
            sample_weight: Optional per-observation weights of ``X``. When
                the holdout is split off ``X``, its part of the weights
                weighs the holdout for early stopping and the leaf refit;
                a given holdout is unweighted.

        Returns:
            ``self``, with ``ensemble_``, ``fit_report_``,
            ``n_features_in_`` and ``feature_names_in_`` (DataFrame input
            only) set.

        Raises:
            ValueError: If ``X`` is not 2-D or only one of ``X_holdout``
                and ``y_holdout`` is given.
        """
        if (X_holdout is None) != (y_holdout is None):
            raise ValueError("give both X_holdout and y_holdout, or neither")
        was_frame = isinstance(X, pd.DataFrame)
        X = _to_frame(X)
        cat_pos = [X.columns.get_loc(c) for c in categorical_columns(X)]
        self.n_features_in_ = X.shape[1]
        if was_frame:
            self.feature_names_in_ = np.asarray(X.columns)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

        loss, target = self._encode(y)
        w_hold = None
        if X_holdout is None:
            train_idx, hold_idx = train_test_split(
                np.arange(len(X)), test_size=self.holdout_fraction,
                random_state=self.random_state, stratify=y if self._classifier else None)
            X_hold, y_hold = X.iloc[hold_idx], target[hold_idx]
            if w is not None:
                w, w_hold = w[train_idx], w[hold_idx]
            X, target = X.iloc[train_idx], target[train_idx]
        else:
            X_hold = _to_frame(X_holdout)
            _, y_hold = self._encode(y_holdout)

        params = dict(_CATBOOST_DEFAULTS, random_seed=self.random_state,
                      early_stopping_rounds=early_stopping_rounds(len(X)), verbose=100 if self.verbose else 0)
        params.update(self.catboost_params or {})
        params["loss_function"] = loss  # fixed by the target, not overridable
        model = (CatBoostClassifier if self._classifier else CatBoostRegressor)(**params)
        model.fit(
            _stringified(X, cat_pos), target, cat_features=cat_pos, sample_weight=w,
            eval_set=Pool(_stringified(X_hold, cat_pos), y_hold, cat_features=cat_pos, weight=w_hold),
        )

        self.ensemble_ = ObliviousTreeEnsemble.from_catboost(model, self.leaf_type)
        self.fit_report_ = self.ensemble_.fit(
            X, target, X_hold, y_hold, strategy=self.strategy, optimizer=self.optimizer,
            sample_weight=w, holdout_weight=w_hold,
            split_budget=self.split_budget, max_bytes=self.max_bytes, verbose=self.verbose,
        )
        return self


class BoxwoodClassifier(ClassifierMixin, _BoxwoodBase):
    """A boxwood classifier: CatBoost builds the trees, boxwood refits the leaves.

    ``fit`` trains a CatBoost classifier with early stopping on the holdout,
    imports the oblivious trees and refits the leaf values with
    :meth:`ObliviousTreeEnsemble.fit`.

    Fitted attributes:
        classes_: The class labels, sorted.
        ensemble_: The refitted :class:`ObliviousTreeEnsemble`.
        fit_report_: The report dict of the leaf refit.
        n_features_in_: The number of input columns.
        feature_names_in_: The column names (DataFrame input only).
    """

    _classifier = True

    def _encode(self, y):
        return ("Logloss" if len(self.classes_) == 2 else "MultiClass"), self._class_index(y)

    def _class_index(self, y) -> np.ndarray:
        """The index of every label of ``y`` in :attr:`classes_`.

        Raises:
            ValueError: If a label is not in :attr:`classes_`.
        """
        arr = np.asarray(y)
        idx = np.searchsorted(self.classes_, arr)
        known = idx < self.classes_.size
        known &= self.classes_[np.where(known, idx, 0)] == arr
        if not known.all():
            raise ValueError(f"label {arr[~known][0]!r} is not in classes_")
        return idx.astype(np.int64)

    def fit(self, X, y, X_holdout=None, y_holdout=None, sample_weight=None):
        """Train CatBoost on the data and refit the leaf values.

        ``classes_ = np.unique(y)``; CatBoost uses ``Logloss`` (2 classes)
        or ``MultiClass`` (more).

        Args:
            X: Training features, a DataFrame or 2-D numpy array.
                Categorical columns are the non-numeric ones plus bool.
            y: The ``n`` training labels.
            X_holdout: Optional holdout features; ``holdout_fraction`` of
                the observations are split off ``X`` when it is None (stratified
                for hard targets).
            y_holdout: The holdout target, same rules as ``y``; required
                with ``X_holdout``.
            sample_weight: Optional per-observation weights of ``X``,
                passed to CatBoost and to the leaf refit.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``X`` is not 2-D or only one of ``X_holdout``
                and ``y_holdout`` is given.
        """
        self.classes_ = np.unique(y)
        return self._fit_common(X, y, X_holdout, y_holdout, sample_weight)

    def update(self, X, y, steps=1, sample_weight=None):
        """Adapt the leaf values to a window of new data.

        The labels are mapped to the class indices with :attr:`classes_`;
        the update then runs on the ensemble (see
        :meth:`ObliviousTreeEnsemble.update`).

        Args:
            X: The window features, with the columns of :meth:`fit`.
            y: The ``n`` class labels; every label must be in
                :attr:`classes_`.
            steps: Number of optimizer iterations; must be >= 1.
            sample_weight: Optional per-observation weights of the window observations.

        Returns:
            The report dict of the ensemble update.

        Raises:
            ValueError: If a label is not in :attr:`classes_`, no
                :meth:`fit` happened before, ``steps`` < 1, or a target
                or weight has the wrong length.
        """
        return self.ensemble_.update(X, self._class_index(y), steps=steps,
                                    sample_weight=sample_weight)

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities for every observation of ``X``.

        Args:
            X: The features, with the columns of :meth:`fit` (see
                :meth:`ObliviousTreeEnsemble.predict`).

        Returns:
            A float64 array of shape ``(n, n_classes)`` whose rows sum to 1.

        Raises:
            ValueError: If ``X`` is not 2-D or its column count differs
                from the one seen in :meth:`fit`.
        """
        p = self.ensemble_.predict(X)
        if p.shape[1] == 1:
            p = p[:, 0]
            return np.column_stack([1.0 - p, p])
        return p

    def predict(self, X) -> np.ndarray:
        """The most likely class label for every observation of ``X``.

        Args:
            X: The features, with the columns of :meth:`fit`.

        Returns:
            An array of ``n`` labels from :attr:`classes_`.

        Raises:
            ValueError: As in :meth:`predict_proba`.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class BoxwoodRegressor(RegressorMixin, _BoxwoodBase):
    """A boxwood regressor: CatBoost builds the trees, boxwood refits the leaves.

    ``fit`` trains a CatBoost regressor (``RMSE``) with early stopping on
    the holdout, imports the oblivious trees and refits the leaf values
    with :meth:`ObliviousTreeEnsemble.fit`.

    Fitted attributes:
        ensemble_: The refitted :class:`ObliviousTreeEnsemble`.
        fit_report_: The report dict of the leaf refit.
        n_features_in_: The number of input columns.
        feature_names_in_: The column names (DataFrame input only).
    """

    def _encode(self, y):
        return "RMSE", np.asarray(y, dtype=np.float64)

    def fit(self, X, y, X_holdout=None, y_holdout=None, sample_weight=None):
        """Train CatBoost on the data and refit the leaf values.

        Args:
            X: Training features, a DataFrame or 2-D numpy array.
                Categorical columns are the non-numeric ones plus bool.
            y: The ``n`` target values.
            X_holdout: Optional holdout features; ``holdout_fraction`` of
                the observations are split off ``X`` when it is None.
            y_holdout: The holdout target; required with ``X_holdout``.
            sample_weight: Optional per-observation weights of ``X``,
                passed to CatBoost and to the leaf refit.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``X`` is not 2-D or only one of ``X_holdout``
                and ``y_holdout`` is given.
        """
        return self._fit_common(X, y, X_holdout, y_holdout, sample_weight)

    def update(self, X, y, steps=1, sample_weight=None):
        """Adapt the leaf values to a window of new data.

        See :meth:`ObliviousTreeEnsemble.update` for what happens on the
        ensemble.

        Args:
            X: The window features, with the columns of :meth:`fit`.
            y: The ``n`` target values.
            steps: Number of optimizer iterations; must be >= 1.
            sample_weight: Optional per-observation weights of the window observations.

        Returns:
            The report dict of the ensemble update.

        Raises:
            ValueError: If no :meth:`fit` happened before, ``steps`` < 1,
                or a target or weight has the wrong length.
        """
        return self.ensemble_.update(X, y, steps=steps, sample_weight=sample_weight)

    def predict(self, X) -> np.ndarray:
        """Predict the target value of every observation of ``X``.

        Args:
            X: The features, with the columns of :meth:`fit` (see
                :meth:`ObliviousTreeEnsemble.predict`).

        Returns:
            A float64 array of ``n`` values.

        Raises:
            ValueError: If ``X`` is not 2-D or its column count differs
                from the one seen in :meth:`fit`.
        """
        return self.ensemble_.predict(X)[:, 0]
