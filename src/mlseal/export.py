"""Export fitted scikit-learn estimators to the JSON model format (training side only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlseal.modelfile import FORMAT, dumps, loads
from mlseal.schema import FEATURE_NAMES

TREE_LEAF = -1


class ExportError(RuntimeError):
    pass


def _tree_arrays(tree: Any) -> dict[str, list]:
    t = tree.tree_
    if t.n_outputs != 1 or t.value.shape[2] != 1:
        raise ExportError("only single-output regression trees are supported")
    leaf = t.children_left == TREE_LEAF
    return {
        "feature": np.where(leaf, -1, t.feature).astype(int).tolist(),
        "threshold": np.where(leaf, 0.0, t.threshold).astype(float).tolist(),
        "left": t.children_left.astype(int).tolist(),
        "right": t.children_right.astype(int).tolist(),
        "value": np.where(leaf, t.value[:, 0, 0], 0.0).astype(float).tolist(),
    }


def _boosting_params(est: Any, X_check: np.ndarray) -> dict[str, Any]:
    if est.n_classes_ != 2 or est.estimators_.shape[1] != 1:
        raise ExportError("only binary gradient boosting is supported")
    if getattr(est, "init", None) not in (None, "zero"):
        raise ExportError("custom init estimators are not supported")
    # Constant initial raw score: the clipped log-odds of the training prior, recovered from
    # the public decision function and cross-checked on several rows.
    stages = sum(est.learning_rate * t.predict(X_check) for t in est.estimators_[:, 0])
    init = est.decision_function(X_check) - stages
    if np.ptp(init) > 1e-9:
        raise ExportError("initial score is not constant; unsupported estimator configuration")
    return {
        "init": float(np.median(init)),
        "learning_rate": float(est.learning_rate),
        "trees": [_tree_arrays(t) for t in est.estimators_[:, 0]],
    }


def to_model_dict(
    est: Any, *, name: str, version: str, threshold: float, X_check: np.ndarray
) -> dict[str, Any]:
    if list(est.classes_) != [0, 1]:
        raise ExportError("expected classes [0, 1] (1 = phishing)")
    kind = type(est).__name__
    if kind == "LogisticRegression":
        if est.coef_.shape != (1, len(FEATURE_NAMES)):
            raise ExportError("unexpected coefficient shape")
        params = {
            "coef": est.coef_[0].astype(float).tolist(),
            "intercept": float(est.intercept_[0]),
        }
        kind_name = "logistic_regression"
    elif kind == "GradientBoostingClassifier":
        params = _boosting_params(est, X_check)
        kind_name = "gradient_boosting"
    else:
        raise ExportError(f"no exporter for {kind}")
    return {
        "format": FORMAT,
        "name": name,
        "version": version,
        "kind": kind_name,
        "features": list(FEATURE_NAMES),
        "positive_class": "phishing",
        "threshold": threshold,
        "params": params,
    }


def export(
    est: Any,
    *,
    name: str,
    version: str,
    threshold: float,
    X_check: np.ndarray,
    tolerance: float = 1e-9,
) -> tuple[bytes, float]:
    """Serialise `est` and prove the JSON runtime reproduces it on `X_check`.

    Returns the file bytes and the largest probability difference observed.
    """
    data = dumps(
        to_model_dict(est, name=name, version=version, threshold=threshold, X_check=X_check)
    )
    expected = est.predict_proba(X_check)[:, 1]
    got = loads(data).predict_proba(X_check)
    max_diff = float(np.max(np.abs(expected - got)))
    if max_diff > tolerance:
        raise ExportError(f"exported model deviates from the estimator by {max_diff:.2e}")
    return data, max_diff
