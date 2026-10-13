"""Quality gates: absolute floors, maximum regression against the approved baseline, and
optionally exact reproduction of the committed report.

The baseline is `reports/metrics.json`, the metrics of the last *promoted* model. Promotion is
a reviewed commit, so a metric change is always visible in a pull request diff.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from mlseal.config import GatePolicy

EXPORT_TOLERANCE = 1e-9
REPRODUCIBILITY_TOLERANCE = 1e-6
# Report fields a faithful retrain on another machine or commit need not match: provenance,
# the software environment, and the model file's exact bytes (tied splits can resolve
# differently in the last bit across operating systems; predictions and metrics still agree).
NOT_REPRODUCED = ("materials", "environment", "model.sha256", "model.bytes")
_MISSING = object()


def _leaves(obj: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _leaves(value, f"{path}.{key}" if path else str(key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _leaves(value, f"{path}[{i}]")
    else:
        yield path, obj


def _compared(path: str) -> bool:
    return not any(path == p or path.startswith((f"{p}.", f"{p}[")) for p in NOT_REPRODUCED)


def _difference(a: Any, b: Any) -> float:
    numbers = all(isinstance(v, int | float) and not isinstance(v, bool) for v in (a, b))
    if numbers:
        delta = abs(a - b)
        return 0.0 if a == b else (math.inf if math.isnan(delta) else float(delta))
    return 0.0 if type(a) is type(b) and a == b else math.inf


def report_drift(fresh: dict[str, Any], committed: dict[str, Any]) -> tuple[float, str]:
    """Largest difference between two metrics reports over every compared value, and where.

    Numbers differ by their absolute difference. A changed string, type or structure (a value
    present on one side only) is an infinite difference.
    """
    a = {path: v for path, v in _leaves(fresh) if _compared(path)}
    b = {path: v for path, v in _leaves(committed) if _compared(path)}
    worst, where = 0.0, ""
    for path in sorted(a.keys() | b.keys()):
        delta = _difference(a.get(path, _MISSING), b.get(path, _MISSING))
        if delta > worst:
            worst, where = delta, path
    return worst, where


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


def run_gates(
    metrics: dict[str, Any],
    policy: GatePolicy,
    baseline: dict[str, Any] | None,
    *,
    expect_reproducible: bool = False,
) -> list[GateResult]:
    test = metrics["test"]["metrics"]
    results = [
        GateResult(
            f"min_{name}",
            test[name] >= floor,
            f"{name} {test[name]:.4f} (floor {floor:.4f})",
        )
        for name, floor in policy.minimum.items()
    ]
    diff = metrics["test"]["export_max_abs_diff"]
    results.append(
        GateResult(
            "export_parity",
            diff <= EXPORT_TOLERANCE,
            f"JSON runtime vs scikit-learn max |Δp| {diff:.1e} (max {EXPORT_TOLERANCE:.0e})",
        )
    )
    if baseline is None:
        results.append(GateResult("regression", True, "no baseline yet: first release"))
        if expect_reproducible:
            results.append(GateResult("reproducible", False, "no committed report to reproduce"))
        return results

    base = baseline["test"]["metrics"]
    for name, max_drop in policy.max_regression.items():
        drop = base[name] - test[name]
        results.append(
            GateResult(
                f"regression_{name}",
                drop <= max_drop,
                f"{name} {test[name]:.4f} vs baseline {base[name]:.4f} "
                f"(drop {drop:+.4f}, max {max_drop:.4f})",
            )
        )
    if expect_reproducible:
        # Every value of the report, not only the headline metrics: intervals, counts, the
        # selection table and the figures' data cannot be stale or edited by hand either.
        delta, where = report_drift(metrics, baseline)
        same_data = (
            metrics["materials"]["dataset"]["sha256"]
            == (baseline["materials"]["dataset"]["sha256"])
        )
        drift = "every value identical" if delta == 0 else f"max |Δ| {delta:.1e} at {where}"
        results.append(
            GateResult(
                "reproducible",
                delta <= REPRODUCIBILITY_TOLERANCE and same_data,
                f"retrain vs the committed report: {drift} "
                f"(tolerance {REPRODUCIBILITY_TOLERANCE:.0e}); same dataset digest: {same_data}",
            )
        )
    return results


def summary(results: list[GateResult], baseline_path: str | None) -> dict[str, Any]:
    return {
        "passed": all(r.passed for r in results),
        "baseline": baseline_path,
        "results": [asdict(r) for r in results],
    }
