"""`mlseal train`: data gate -> de-duplicated split -> CV selection -> evaluation -> export.

Everything the release needs lands in one bundle directory:

    model.json            safe JSON model (the only file the server loads)
    metrics.json          evaluation, selection and duplicate analysis, input digests
    model_card.md         human-readable summary generated from metrics.json
    data_validation.json  data-gate report, including the train/test overlap check

`mlseal gate` adds gates.json and `mlseal sign` adds the signed manifest.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from mlseal import __version__
from mlseal.config import LogisticSpec, PipelineConfig
from mlseal.data import deduplicate, parse_arff, sha256_bytes, sha256_file
from mlseal.export import export
from mlseal.metrics import binary_metrics, bootstrap_ci, confusion, curves
from mlseal.modelcard import render_model_card
from mlseal.provenance import git_source
from mlseal.training import describe, make_estimator, protocol_comparison, select_model
from mlseal.validation import (
    DataValidationError,
    load_reference,
    split_check,
    validate_bytes,
)

METRICS_SCHEMA = "mlseal.metrics/v1"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "mlseal": __version__,
        **{pkg: metadata.version(pkg) for pkg in ("numpy", "scipy", "scikit-learn")},
    }


def materials(cfg: PipelineConfig, config_path: Path, lock_path: Path) -> dict[str, Any]:
    """Digests of every input that determines the model (recorded at training time)."""
    return {
        "dataset": {"path": str(cfg.data.path), "sha256": sha256_file(cfg.data.path)},
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "lockfile": {
            "path": str(lock_path),
            "sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        },
        "source": git_source(Path.cwd()),
    }


def train(
    cfg: PipelineConfig,
    bundle_dir: Path,
    *,
    config_path: Path,
    lock_path: Path = Path("uv.lock"),
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    seed, thr = cfg.seed, cfg.evaluation.threshold

    # 1. Data gate: integrity, schema, domains, balance, duplicates, conflicts, drift.
    raw = cfg.data.path.read_bytes()
    reference = load_reference(cfg.data.reference_profile)
    report = validate_bytes(raw, cfg.data.policy, reference, path=str(cfg.data.path))
    if report.passed:
        log(f"data gate passed ({len(report.checks)} checks)")
    ds = parse_arff(raw) if report.passed else None

    # 2. One row per distinct feature vector, then a stratified hold-out split.
    if ds is not None:
        X, y = ds.X, ds.y
        Xu, yu, stats = deduplicate(X, y)
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xu, yu, test_size=cfg.split.test_size, stratify=yu, random_state=seed
        )
        report.checks.append(split_check(X_tr, X_te))
    write_json(bundle_dir / "data_validation.json", report.to_dict())
    if not report.passed:
        raise DataValidationError(report)
    log(
        f"{stats.rows} rows -> {stats.unique_vectors} distinct vectors "
        f"({stats.duplicate_share:.1%} duplicate rows); train {len(y_tr)}, test {len(y_te)}"
    )

    # 3. Model selection by cross-validation on the training split only.
    best, candidates = select_model(
        X_tr, y_tr, cfg.model.candidates, cfg.split.cv_folds, cfg.model.selection_metric, seed
    )
    spec = cfg.model.candidates[best]
    log(f"selected {describe(spec)} (CV {cfg.model.selection_metric}, one-standard-error rule)")

    # 4. Fit on the training split; the shipped model is exactly the evaluated model.
    est = make_estimator(spec, seed).fit(X_tr, y_tr)
    p = est.predict_proba(X_te)[:, 1]
    test = binary_metrics(y_te, p, thr)
    ci = bootstrap_ci(y_te, p, thr, cfg.evaluation.bootstrap, seed)

    reference_model: dict[str, Any] | None = None
    ref_spec = next((c for c in cfg.model.candidates if isinstance(c, LogisticSpec)), None)
    curve_data = {"selected": curves(y_te, p)}
    if ref_spec is not None and ref_spec is not spec:
        p_ref = make_estimator(ref_spec, seed).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        reference_model = {
            "description": describe(ref_spec),
            "metrics": binary_metrics(y_te, p_ref, thr),
        }
        curve_data["reference"] = curves(y_te, p_ref)

    # 5. How much would duplicates have inflated the estimate?
    log("comparing evaluation protocols (naive / grouped / de-duplicated)")
    effect = protocol_comparison(X, y, spec, cfg.split.cv_folds, seed)

    # 6. Export to the safe format and prove the JSON runtime matches scikit-learn.
    model_bytes, parity = export(
        est, name=cfg.model.name, version=cfg.model.version, threshold=thr, X_check=Xu
    )
    (bundle_dir / "model.json").write_bytes(model_bytes)

    metrics = {
        "schema": METRICS_SCHEMA,
        "model": {
            "name": cfg.model.name,
            "version": cfg.model.version,
            "description": describe(spec),
            "candidate": spec.model_dump(),
            "sha256": sha256_bytes(model_bytes),
            "bytes": len(model_bytes),
        },
        "data": {
            "rows": stats.rows,
            "unique_vectors": stats.unique_vectors,
            "duplicate_rows": stats.duplicate_rows,
            "duplicate_share": stats.duplicate_share,
            "conflicting_vectors": stats.conflicting_vectors,
            "conflicting_rows": stats.conflicting_rows,
            "ties_dropped": stats.ties_dropped,
            "deduplicated_rows": len(yu),
            "deduplicated_phishing": stats.phishing,
            "deduplicated_legitimate": stats.legitimate,
            "raw_phishing_share": float(y.mean()),
            "train_rows": len(y_tr),
            "test_rows": len(y_te),
            "test_phishing": int(y_te.sum()),
        },
        "protocol": {
            "evaluation": "stratified hold-out on de-duplicated feature vectors",
            "test_size": cfg.split.test_size,
            "cv_folds": cfg.split.cv_folds,
            "seed": seed,
            "threshold": thr,
            "bootstrap_resamples": cfg.evaluation.bootstrap,
        },
        "selection": {
            "metric": cfg.model.selection_metric,
            "rule": "simplest candidate within one standard error of the best CV mean",
            "selected": best,
            "candidates": candidates,
        },
        "test": {
            "metrics": test,
            "ci95": ci,
            "confusion": confusion(y_te, p, thr),
            "export_max_abs_diff": parity,
        },
        "reference_model": reference_model,
        "duplicates_effect": effect,
        "curves": curve_data,
        "materials": materials(cfg, config_path, lock_path),
        "environment": environment(),
    }
    write_json(bundle_dir / "metrics.json", metrics)
    (bundle_dir / "model_card.md").write_text(
        render_model_card(metrics, report.to_dict()), encoding="utf-8"
    )
    log(
        f"test ROC-AUC {test['roc_auc']:.4f}, F1 {test['f1']:.4f}, "
        f"accuracy {test['accuracy']:.4f} (n={len(y_te)})"
    )
    return metrics
