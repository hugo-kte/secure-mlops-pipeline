from pathlib import Path

import numpy as np
import pytest

from mlseal.metrics import _batch_metrics, binary_metrics, bootstrap_ci, curves, t_interval


@pytest.mark.parametrize("seed", range(20))
def test_vectorised_metrics_match_scikit_learn_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 400))
    y = (rng.random(n) < rng.uniform(0.2, 0.8)).astype(np.int64)
    y[:2] = [0, 1]
    p = np.round(rng.random(n), int(rng.integers(1, 4)))  # coarse rounding -> many ties
    batch = {k: v[0] for k, v in _batch_metrics(y[None], p[None], 0.5).items()}
    for name, value in binary_metrics(y, p, 0.5).items():
        assert batch[name] == pytest.approx(value, abs=1e-12), name


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(800) < 0.5).astype(int)
    p = np.clip(y * 0.6 + rng.random(800) * 0.5, 0, 1)
    point = binary_metrics(y, p, 0.5)
    ci = bootstrap_ci(y, p, 0.5, n_boot=500, seed=1)
    for name, (lo, hi) in ci.items():
        assert lo <= point[name] <= hi, name
        assert hi - lo < 0.2


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.5).astype(int)
    p = rng.random(300)
    assert bootstrap_ci(y, p, 0.5, 300, seed=7) == bootstrap_ci(y, p, 0.5, 300, seed=7)


def test_t_interval():
    out = t_interval([0.9, 0.92, 0.91, 0.93, 0.89])
    assert out["mean"] == pytest.approx(0.91)
    assert out["ci95"][0] < 0.91 < out["ci95"][1]


def test_curves_are_thinned_and_monotone():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.5).astype(int)
    c = curves(y, rng.random(2000), points=50)
    assert len(c["roc"]["fpr"]) <= 50
    assert np.all(np.diff(c["roc"]["fpr"]) >= 0)
    assert np.all(np.diff(c["pr"]["recall"]) >= 0)


def test_grouped_folds_keep_identical_vectors_together():
    from mlseal.data import group_ids, read_arff
    from mlseal.training import grouped_folds

    ds = read_arff(Path(__file__).resolve().parents[1] / "data" / "phishing_websites.arff")
    gid = group_ids(ds.X)
    folds = grouped_folds(ds.X, ds.y, 5, seed=0)
    assert sorted(np.concatenate([te for _, te in folds]).tolist()) == list(range(len(gid)))
    for train_idx, test_idx in folds:
        assert not set(gid[train_idx]) & set(gid[test_idx])
        assert abs(ds.y[test_idx].mean() - ds.y.mean()) < 0.03  # roughly stratified
