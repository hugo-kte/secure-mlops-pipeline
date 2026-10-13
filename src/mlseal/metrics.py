"""Classification metrics with uncertainty: percentile bootstrap and t-intervals over folds."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")


def confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, int]:
    pred = p >= threshold
    yb = y.astype(bool)
    return {
        "tp": int((pred & yb).sum()),
        "fp": int((pred & ~yb).sum()),
        "tn": int((~pred & ~yb).sum()),
        "fn": int((~pred & yb).sum()),
    }


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    c = confusion(y, p, threshold)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {
        "accuracy": (tp + c["tn"]) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


def _batch_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    """`binary_metrics` for a batch of resamples at once (rows of y and p).

    ROC-AUC is the Mann-Whitney statistic with average ranks for ties; average precision
    sums precision at the end of each run of tied scores, exactly as scikit-learn does.
    """
    pos = y.sum(axis=1)
    neg = y.shape[1] - pos
    pred = p >= threshold
    tp = (pred & (y == 1)).sum(axis=1)
    fp = (pred & (y == 0)).sum(axis=1)
    fn = pos - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "accuracy": 1.0 - (fp + fn) / y.shape[1],
            "precision": np.where(tp + fp > 0, tp / (tp + fp), 0.0),
            "recall": np.where(pos > 0, tp / pos, 0.0),
            "f1": np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0),
        }
        ranks = stats.rankdata(p, axis=1)
        out["roc_auc"] = ((ranks * y).sum(axis=1) - pos * (pos + 1) / 2) / (pos * neg)

        order = np.argsort(-p, axis=1, kind="stable")
        ps = np.take_along_axis(p, order, axis=1)
        ys = np.take_along_axis(y, order, axis=1)
        precision_at = np.cumsum(ys, axis=1) / np.arange(1, y.shape[1] + 1)
        idx = np.arange(y.shape[1])
        is_end = np.ones_like(ps, dtype=bool)
        is_end[:, :-1] = ps[:, :-1] != ps[:, 1:]
        end = np.minimum.accumulate(np.where(is_end, idx, y.shape[1])[:, ::-1], axis=1)[:, ::-1]
        out["average_precision"] = (ys * np.take_along_axis(precision_at, end, axis=1)).sum(
            axis=1
        ) / pos
    return out


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
    level: float = 0.95,
    batch: int = 500,
) -> dict[str, list[float]]:
    """Percentile bootstrap over test rows (rows are distinct feature vectors, so i.i.d.)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    samples: dict[str, list[np.ndarray]] = {m: [] for m in METRICS}
    for start in range(0, n_boot, batch):
        idx = rng.integers(0, n, (min(batch, n_boot - start), n))
        yb, pb = y[idx].astype(np.int64), p[idx]
        both = (yb.min(axis=1) == 0) & (yb.max(axis=1) == 1)  # AUC needs both classes
        for name, values in _batch_metrics(yb[both], pb[both], threshold).items():
            samples[name].append(values)
    alpha = (1.0 - level) / 2.0
    out = {}
    for name, chunks in samples.items():
        values = np.concatenate(chunks)
        out[name] = [float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))]
    return out


def t_interval(scores: list[float], level: float = 0.95) -> dict[str, Any]:
    """Mean and t-interval over CV folds (folds are not independent: read it as approximate)."""
    arr = np.asarray(scores, dtype=float)
    mean = float(arr.mean())
    t = stats.t.ppf(0.5 + level / 2.0, len(arr) - 1)
    half = float(t * arr.std(ddof=1) / np.sqrt(len(arr)))
    return {"mean": mean, "ci95": [mean - half, mean + half], "folds": [float(s) for s in arr]}


def curves(y: np.ndarray, p: np.ndarray, points: int = 200) -> dict[str, dict[str, list[float]]]:
    """ROC and precision-recall curves, thinned to at most `points` points for the report."""

    def thin(*arrays: np.ndarray) -> list[list[float]]:
        n = len(arrays[0])
        idx = np.unique(np.linspace(0, n - 1, min(points, n)).round().astype(int))
        return [[round(float(v), 5) for v in a[idx]] for a in arrays]

    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    fpr_t, tpr_t = thin(fpr, tpr)
    rec_t, prec_t = thin(recall[::-1], precision[::-1])
    return {"roc": {"fpr": fpr_t, "tpr": tpr_t}, "pr": {"recall": rec_t, "precision": prec_t}}
