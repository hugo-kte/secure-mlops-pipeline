"""Policy as code: properties of the repository itself that must not silently regress."""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from conftest import ROOT
from mlseal.modelcard import render_model_card

SRC = ROOT / "src" / "mlseal"
# Modules that load or deserialise objects (directly or through pickle) and the heavy
# training stack: none of them may be reachable from the serving path.
FORBIDDEN = {
    "pickle", "_pickle", "cPickle", "dill", "cloudpickle", "joblib", "skops", "shelve",
    "marshal", "sklearn", "torch", "pandas",
}  # fmt: skip
SERVING_MODULES = [
    *sorted((SRC / "serving").glob("*.py")),
    SRC / "modelfile.py",
    SRC / "signing.py",
    SRC / "provenance.py",
    SRC / "schema.py",
    SRC / "data.py",
]


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".")[0]


@pytest.mark.parametrize("path", SERVING_MODULES, ids=lambda p: p.name)
def test_serving_path_never_imports_a_deserialiser(path):
    assert not set(_imports(path)) & FORBIDDEN


def test_serving_app_does_not_load_the_training_stack():
    code = (
        "import sys, mlseal.serving.app, mlseal.cli; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & "
        "{'sklearn', 'scipy', 'matplotlib', 'joblib', 'pandas', 'yaml'}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""},
    )
    assert out.stdout.strip() == "[]"


# --- CI workflow hygiene -------------------------------------------------------------------

WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")


def _load(path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    return doc, doc.get("on", doc.get(True))  # PyYAML reads the key `on` as True


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_is_least_privilege_and_pinned(path):
    doc, triggers = _load(path)
    assert "pull_request_target" not in triggers
    assert doc["permissions"] == {}, "deny by default at workflow level"
    for name, job in doc["jobs"].items():
        assert "permissions" in job, f"job {name} must declare its permissions"
        assert not job["runs-on"].endswith("-latest"), f"job {name}: pin the runner image"
        for step in job["steps"]:
            uses = step.get("uses")
            if uses is None:
                assert "${{ github.event" not in step.get("run", ""), "template injection"
                continue
            assert PINNED.match(uses), f"{uses} is not pinned to a commit SHA"
            if uses.startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_only_the_attestation_job_can_mint_tokens(path):
    doc, _ = _load(path)
    writers = {
        name
        for name, job in doc["jobs"].items()
        if any(v == "write" for v in job["permissions"].values())
    }
    assert writers <= {"attest"}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_provenance_is_attested_only_for_fully_green_builds(path):
    doc, _ = _load(path)
    for name, job in doc["jobs"].items():
        if job["permissions"].get("attestations") == "write":
            needs = job.get("needs", [])
            needs = {needs} if isinstance(needs, str) else set(needs)
            assert needs == set(doc["jobs"]) - {name}, f"{name} must wait for every other job"


# --- container ------------------------------------------------------------------------------


def test_dockerfile_hardening():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"python:3\.\d+-slim@sha256:[0-9a-f]{64}", text), "base image digest pin"
    assert "--require-hashes" in text and "--only-binary=:all:" in text
    user = re.findall(r"^USER\s+(\S+)", text, flags=re.M)
    assert user and user[-1].split(":")[0] not in ("root", "0")
    assert "HEALTHCHECK" in text
    assert not re.search(r"^ADD\s+https?://", text, flags=re.M)


def test_docker_build_context_is_an_allowlist():
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").split("\n")
    assert lines[[ln.strip() for ln in lines].index("*")] == "*"


# --- repository -----------------------------------------------------------------------------


def test_gitignore_is_anchored():
    lines = [ln.strip() for ln in (ROOT / ".gitignore").read_text().splitlines()]
    for directory in ("build", "runtime", "logs"):
        assert f"/{directory}/" in lines
    assert "data/" not in lines and "data" not in lines  # the dataset is committed
    assert "*.key" in lines


def test_committed_model_card_is_generated_from_the_committed_reports():
    """The CI gate re-derives reports/metrics.json; the card must follow from it untouched."""
    reports = ROOT / "reports"
    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    validation = json.loads((reports / "data_validation.json").read_text(encoding="utf-8"))
    card = (reports / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert card == render_model_card(metrics, validation)


def test_no_private_key_in_the_source_tree():
    pattern = re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")
    skip = {".venv", ".git", "build", "runtime", ".pytest_cache", ".ruff_cache", "__pycache__"}
    for directory, subdirs, files in os.walk(ROOT):
        subdirs[:] = [d for d in subdirs if d not in skip]
        for name in files:
            path = Path(directory) / name
            assert not pattern.search(path.read_bytes()), path
