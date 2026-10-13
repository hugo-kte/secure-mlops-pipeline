"""Release manifest: an in-toto v1 Statement listing every bundle file and its inputs.

The statement is what gets signed (see `signing.py`). Subjects are the bundle files with
their SHA-256; the predicate records the materials the model was built from (dataset, lock
file, configuration, source commit), the builder, the evaluation and the gate outcomes. Each
outcome is read from the bundle's own reports or, for the pickle scan, computed here: no
caller can assert a result that was not produced.
The envelope and statement formats are the ones Sigstore, SLSA and OpenSSF Model Signing
use, so standard tooling can read them; the predicate itself is project-specific.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlseal.data import sha256_file

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = "https://github.com/hugo-kte/secure-mlops-pipeline/model-release/v1"
REPOSITORY = "https://github.com/hugo-kte/secure-mlops-pipeline"
BUNDLE_FILES = ("model.json", "metrics.json", "model_card.md", "data_validation.json", "gates.json")


class ProvenanceError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    # Fixed argument list, no shell: nothing user-controlled reaches the command line.
    proc = subprocess.run(  # noqa: S603
        [git, "-C", str(root), *args], capture_output=True, text=True, check=False, timeout=10
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_source(root: Path) -> dict[str, Any]:
    """Commit of the working tree, and whether it had uncommitted changes."""
    commit = os.environ.get("GITHUB_SHA") or _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "repository": REPOSITORY,
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def builder() -> dict[str, Any]:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        run = os.environ.get("GITHUB_RUN_ID", "")
        return {
            "id": "github-actions",
            "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
            "run": f"{server}/{repo}/actions/runs/{run}",
        }
    return {"id": "local"}


def _check_material(name: str, recorded: dict[str, Any]) -> dict[str, Any]:
    """The file must still hash to what training recorded, or the manifest would lie."""
    path = Path(recorded["path"])
    if recorded["sha256"] is None:
        return recorded
    if not path.is_file() or sha256_file(path) != recorded["sha256"]:
        raise ProvenanceError(f"{name} changed since training: {path}")
    return recorded


def _pickle_scan(bundle_dir: Path) -> str:
    """Scan the bundle here, so the signed outcome can only describe a scan that ran on it."""
    from mlseal.scanner import scan_path  # lazy: the server imports this module, not scans

    refused = [r.path for r in scan_path(bundle_dir) if r.is_pickle]
    if refused:
        raise ProvenanceError(f"pickles, or files the scanner refused, in the bundle: {refused}")
    return "clean"


def build_statement(bundle_dir: Path) -> dict[str, Any]:
    missing = [f for f in BUNDLE_FILES if not (bundle_dir / f).is_file()]
    if missing:
        raise ProvenanceError(f"bundle is incomplete, missing {missing}")
    pickle_scan = _pickle_scan(bundle_dir)
    metrics = json.loads((bundle_dir / "metrics.json").read_text(encoding="utf-8"))
    gates = json.loads((bundle_dir / "gates.json").read_text(encoding="utf-8"))
    validation = json.loads((bundle_dir / "data_validation.json").read_text(encoding="utf-8"))
    mats = metrics["materials"]
    subjects = [
        {"name": f, "digest": {"sha256": sha256_file(bundle_dir / f)}} for f in BUNDLE_FILES
    ]
    if subjects[0]["digest"]["sha256"] != metrics["model"]["sha256"]:
        raise ProvenanceError("model.json does not match the model digest in metrics.json")
    return {
        "_type": STATEMENT_TYPE,
        "subject": subjects,
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "model": {k: metrics["model"][k] for k in ("name", "version", "description", "sha256")},
            "materials": [
                {"name": "dataset", **_check_material("dataset", mats["dataset"])},
                {"name": "config", **_check_material("config", mats["config"])},
                {"name": "lockfile", **_check_material("lockfile", mats["lockfile"])},
            ],
            "source": mats["source"],
            "builder": builder(),
            "evaluation": {
                "protocol": metrics["protocol"]["evaluation"],
                "test_rows": metrics["data"]["test_rows"],
                "metrics": metrics["test"]["metrics"],
                "ci95": metrics["test"]["ci95"],
            },
            "policy": {
                "data_validation": "passed" if validation["passed"] else "failed",
                "quality_gates": "passed" if gates["passed"] else "failed",
                "pickle_scan": pickle_scan,
            },
            "environment": metrics["environment"],
            "signed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }
