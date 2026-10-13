"""Model candidates, cross-validated selection and the duplicate-aware evaluation protocols."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from mlseal.config import BoostingSpec, CandidateSpec, LogisticSpec
from mlseal.data import deduplicate, group_ids
from mlseal.metrics import binary_metrics, t_interval

Splits = list[tuple[np.ndarray, np.ndarray]]


def n_jobs() -> int:
    """Worker processes for CV fits: MLSEAL_N_JOBS, else up to 4 (results do not depend on it)."""
    requested = int(os.environ.get("MLSEAL_N_JOBS", "0"))
    return max(1, min(requested or 4, os.cpu_count() or 1, 8))


def make_estimator(spec: CandidateSpec, seed: int) -> Any:
    if isinstance(spec, LogisticSpec):
        return LogisticRegression(C=spec.C, max_iter=5000)
    if isinstance(spec, BoostingSpec):
        return GradientBoostingClassifier(
            n_estimators=spec.n_estimators,
            max_depth=spec.max_depth,
            learning_rate=spec.learning_rate,
            subsample=spec.subsample,
            random_state=seed,
        )
    raise TypeError(f"unknown candidate {spec!r}")


def describe(spec: CandidateSpec) -> str:
    if isinstance(spec, LogisticSpec):
        return f"logistic regression (C={spec.C:g})"
    return (
        f"gradient boosting ({spec.n_estimators} trees, depth {spec.max_depth}, "
        f"learning rate {spec.learning_rate:g})"
    )


def _fit_predict(
    est: Any, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> np.ndarray:
    return clone(est).fit(X[train_idx], y[train_idx]).predict_proba(X[test_idx])[:, 1]


def _cross_validate(
    jobs: list[tuple[Any, np.ndarray, np.ndarray, Splits]],
) -> list[dict[str, dict[str, Any]]]:
    """Run every (estimator, fold) fit of several CV experiments in one worker pool."""
    tasks = [
        (i, est, X, y, tr, te) for i, (est, X, y, splits) in enumerate(jobs) for tr, te in splits
    ]
    probas = Parallel(n_jobs=n_jobs())(
        delayed(_fit_predict)(est, X, y, tr, te) for _, est, X, y, tr, te in tasks
    )
    scores: list[dict[str, list[float]]] = [{} for _ in jobs]
    for (i, _, _, y, _, te), p in zip(tasks, probas, strict=True):
        for name, value in binary_metrics(y[te], p, 0.5).items():
            scores[i].setdefault(name, []).append(value)
    return [{name: t_interval(v) for name, v in s.items()} for s in scores]


def select_model(
    X: np.ndarray,
    y: np.ndarray,
    candidates: list[CandidateSpec],
    folds: int,
    metric: str,
    seed: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Stratified k-fold CV on the training split only, then the one-standard-error rule.

    Candidates are listed from simplest to most complex; the first whose CV mean is within
    one standard error of the best mean wins, so a negligible gain never buys complexity.
    """
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
    cvs = _cross_validate([(make_estimator(spec, seed), X, y, splits) for spec in candidates])
    results = [
        {"candidate": spec.model_dump(), "description": describe(spec), "cv": cv}
        for spec, cv in zip(candidates, cvs, strict=True)
    ]
    best = max(range(len(results)), key=lambda i: results[i]["cv"][metric]["mean"])
    folds_best = np.asarray(results[best]["cv"][metric]["folds"])
    se = float(folds_best.std(ddof=1) / np.sqrt(len(folds_best)))
    floor = results[best]["cv"][metric]["mean"] - se
    chosen = next(i for i, r in enumerate(results) if r["cv"][metric]["mean"] >= floor)
    for i, r in enumerate(results):
        r["best_mean"] = i == best
        r["within_one_se"] = r["cv"][metric]["mean"] >= floor
    return chosen, results


def grouped_folds(
    X: np.ndarray, y: np.ndarray, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold over distinct feature vectors, mapped back to rows.

    Every copy of a vector lands in the same fold; folds are stratified on each vector's
    majority label. (scikit-learn's StratifiedGroupKFold does the same job with a per-group
    Python loop that takes seconds on 5,785 groups.)
    """
    gid = group_ids(X)
    n_groups = int(gid.max()) + 1
    positives = np.bincount(gid, weights=y, minlength=n_groups)
    group_label = (2 * positives >= np.bincount(gid, minlength=n_groups)).astype(int)
    group_fold = np.empty(n_groups, dtype=np.intp)
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for k, (_, test_groups) in enumerate(splitter.split(np.zeros(n_groups), group_label)):
        group_fold[test_groups] = k
    row_fold = group_fold[gid]
    return [(np.flatnonzero(row_fold != k), np.flatnonzero(row_fold == k)) for k in range(folds)]


def protocol_comparison(
    X: np.ndarray, y: np.ndarray, spec: CandidateSpec, folds: int, seed: int
) -> dict[str, Any]:
    """How much do duplicate feature vectors inflate the estimate? Same model, three protocols.

    * naive: row-level stratified k-fold on all rows; copies of a vector land on both sides.
    * grouped: all rows, but identical vectors always share a fold (`grouped_folds`).
    * deduplicated: one row per distinct vector (majority label), stratified k-fold.
    """
    est = make_estimator(spec, seed)
    gid = group_ids(X)
    naive = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
    seen = [np.isin(gid[te], gid[tr]).mean() for tr, te in naive]
    grouped = grouped_folds(X, y, folds, seed)
    Xu, yu, _ = deduplicate(X, y)
    dedup = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xu, yu))
    cv_naive, cv_grouped, cv_dedup = _cross_validate(
        [(est, X, y, naive), (est, X, y, grouped), (est, Xu, yu, dedup)]
    )
    return {
        "naive": {**cv_naive, "rows": len(y), "test_rows_seen_in_train": float(np.mean(seen))},
        "grouped": {**cv_grouped, "rows": len(y), "test_rows_seen_in_train": 0.0},
        "deduplicated": {**cv_dedup, "rows": len(yu), "test_rows_seen_in_train": 0.0},
    }
