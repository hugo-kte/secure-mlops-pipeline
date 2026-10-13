"""Data-validation gate: nothing is trained on data that fails these checks.

Each check is a small, named predicate with the observed value and the limit, so the JSON
report explains a failure without re-running anything. The gate covers the data poisoning
and supply-chain risks that are cheap to detect mechanically: a swapped or altered file
(hash pin), schema drift, out-of-domain values, label flipping at scale (class balance,
contradictory duplicates) and distribution shift against a reviewed reference profile.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mlseal.config import DataPolicy
from mlseal.data import DataError, Dataset, deduplicate, overlap, parse_arff, sha256_bytes
from mlseal.schema import FEATURES, declared_attributes

PSI_EPS = 1e-4


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str
    observed: Any = None
    limit: Any = None


@dataclass
class ValidationReport:
    path: str
    sha256: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
        }


class DataValidationError(RuntimeError):
    def __init__(self, report: ValidationReport) -> None:
        names = ", ".join(c.name for c in report.failures())
        super().__init__(f"data validation failed: {names}")
        self.report = report


# --- distribution profile -----------------------------------------------------------------


def profile(X: np.ndarray) -> dict[str, dict[str, float]]:
    """Share of each allowed value, per feature (the reference for the drift check)."""
    return {
        f.name: {str(v): round(float(np.mean(X[:, j] == v)), 6) for v in f.values}
        for j, f in enumerate(FEATURES)
    }


def psi(expected: dict[str, float], actual: dict[str, float]) -> float:
    """Population stability index between two categorical distributions."""
    total = 0.0
    for value in expected.keys() | actual.keys():
        e = max(expected.get(value, 0.0), PSI_EPS)
        a = max(actual.get(value, 0.0), PSI_EPS)
        total += (a - e) * np.log(a / e)
    return float(total)


# --- checks -------------------------------------------------------------------------------


def _schema(ds: Dataset) -> Check:
    expected = declared_attributes()
    got = tuple((name, tuple(sorted(dom))) for name, dom in ds.attributes)
    want = tuple((name, tuple(sorted(dom))) for name, dom in expected)
    if got == want:
        return Check("schema", True, f"{len(want)} attributes match the contract")
    missing = [n for n, _ in want if n not in dict(got)]
    extra = [n for n, _ in got if n not in dict(want)]
    changed = [n for (n, d) in got if n in dict(want) and dict(want)[n] != d]
    order = [n for n, _ in got] != [n for n, _ in want]
    return Check(
        "schema",
        False,
        f"missing={missing} extra={extra} domain_changed={changed} order_changed={order}",
    )


def _domains(ds: Dataset) -> Check:
    bad: dict[str, int] = {}
    for j, (name, allowed) in enumerate(declared_attributes()):
        if j >= ds.values.shape[1]:
            break
        n_bad = int((~np.isin(ds.values[:, j], allowed)).sum())
        if n_bad:
            bad[name] = n_bad
    if bad:
        return Check("allowed_values", False, f"out-of-domain values: {bad}", observed=bad)
    return Check("allowed_values", True, "every value is in its declared domain", observed=0)


def _between(name: str, value: float, lo: float, hi: float, what: str) -> Check:
    ok = lo <= value <= hi
    return Check(name, ok, f"{what} {value:g} (allowed {lo:g}..{hi:g})", value, [lo, hi])


def _at_most(name: str, value: float, limit: float, what: str) -> Check:
    return Check(name, value <= limit, f"{what} {value:.4f} (max {limit:g})", value, limit)


def validate_bytes(
    raw: bytes,
    policy: DataPolicy,
    reference: dict[str, dict[str, float]] | None,
    *,
    path: str = "<memory>",
) -> ValidationReport:
    digest = sha256_bytes(raw)
    report = ValidationReport(path=path, sha256=digest)
    report.checks.append(
        Check(
            "integrity",
            digest == policy.sha256,
            "SHA-256 matches the pinned digest"
            if digest == policy.sha256
            else f"SHA-256 {digest} does not match the pinned {policy.sha256}",
        )
    )
    try:
        ds = parse_arff(raw)
    except DataError as exc:
        report.checks.append(Check("parse", False, str(exc)))
        return report
    report.checks.append(Check("parse", True, f"{len(ds)} rows, {len(ds.attributes)} columns"))

    schema = _schema(ds)
    report.checks.append(schema)
    if not schema.passed:  # the remaining checks assume the contract's column layout
        return report
    domains = _domains(ds)
    report.checks.append(domains)
    if not domains.passed:
        return report

    X, y = ds.X, ds.y
    report.checks.append(_between("row_count", len(ds), policy.min_rows, policy.max_rows, "rows:"))
    lo, hi = policy.phishing_share
    report.checks.append(_between("label_balance", float(y.mean()), lo, hi, "phishing share:"))
    _, _, stats = deduplicate(X, y)
    report.checks.append(
        _at_most("duplicates", stats.duplicate_share, policy.max_duplicate_share, "duplicate rows:")
    )
    report.checks.append(
        _at_most(
            "label_conflicts",
            stats.conflict_share,
            policy.max_conflict_share,
            f"{stats.conflicting_vectors} vectors carry both labels, share:",
        )
    )
    if reference is not None:
        current = profile(X)
        drift = {name: psi(reference[name], current[name]) for name in reference}
        worst = max(drift, key=drift.__getitem__)
        report.checks.append(
            _at_most(
                "feature_drift",
                drift[worst],
                policy.max_feature_psi,
                f"max PSI vs reference ({worst}):",
            )
        )
    return report


def validate_file(
    path: Path, policy: DataPolicy, reference: dict[str, dict[str, float]] | None
) -> ValidationReport:
    return validate_bytes(Path(path).read_bytes(), policy, reference, path=str(path))


def split_check(X_train: np.ndarray, X_test: np.ndarray) -> Check:
    """No feature vector may appear on both sides of the train/test split."""
    shared = overlap(X_train, X_test)
    return Check(
        "train_test_overlap",
        shared == 0,
        f"{shared} feature vectors appear in both train and test",
        shared,
        0,
    )


def load_reference(path: Path) -> dict[str, dict[str, float]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
