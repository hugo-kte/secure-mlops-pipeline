"""Quality gates: absolute floors, maximum regression against the approved baseline, and
optionally exact reproduction of the committed report.

The baseline is `reports/metrics.json`, the metrics of the last *promoted* model. Promotion is
a reviewed commit, so a metric change is always visible in a pull request diff.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mlseal.config import GatePolicy
from mlseal.metrics import METRICS

EXPORT_TOLERANCE = 1e-9
REPRODUCIBILITY_TOLERANCE = 1e-6


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
        worst = max(METRICS, key=lambda m: abs(test[m] - base[m]))
        delta = abs(test[worst] - base[worst])
        same_data = (
            metrics["materials"]["dataset"]["sha256"]
            == (baseline["materials"]["dataset"]["sha256"])
        )
        results.append(
            GateResult(
                "reproducible",
                delta <= REPRODUCIBILITY_TOLERANCE and same_data,
                f"retrained metrics match the committed report "
                f"(max |Δ| {delta:.1e} on {worst}, same dataset digest: {same_data})",
            )
        )
    return results


def summary(results: list[GateResult], baseline_path: str | None) -> dict[str, Any]:
    return {
        "passed": all(r.passed for r in results),
        "baseline": baseline_path,
        "results": [asdict(r) for r in results],
    }
