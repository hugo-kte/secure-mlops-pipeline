import copy
import json
import shutil

import pytest

from conftest import DATA, ROOT
from mlseal.config import load_config
from mlseal.data import sha256_file
from mlseal.gates import run_gates
from mlseal.provenance import (
    BUNDLE_FILES,
    PREDICATE_TYPE,
    STATEMENT_TYPE,
    ProvenanceError,
    build_statement,
    git_source,
)
from mlseal.signing import load_public_key, verify_bundle


def test_statement_lists_every_bundle_file_and_the_inputs(signed_bundle, keys, trusted_cfg):
    verified = verify_bundle(signed_bundle, [load_public_key(keys.public.read_bytes())])
    s = verified.statement
    assert s["_type"] == STATEMENT_TYPE
    assert s["predicateType"] == PREDICATE_TYPE
    assert [x["name"] for x in s["subject"]] == list(BUNDLE_FILES)
    for subject in s["subject"]:
        assert subject["digest"]["sha256"] == sha256_file(signed_bundle / subject["name"])
    materials = {m["name"]: m for m in s["predicate"]["materials"]}
    assert materials["dataset"]["sha256"] == trusted_cfg.data.policy.sha256
    assert materials["lockfile"]["sha256"] == sha256_file(ROOT / "uv.lock")
    assert s["predicate"]["policy"] == {
        "data_validation": "passed", "quality_gates": "passed", "pickle_scan": "clean",
    }  # fmt: skip


@pytest.fixture(scope="module")
def trusted_cfg():
    return load_config(ROOT / "configs" / "pipeline.yaml")


def test_statement_refuses_inputs_changed_since_training(tmp_path, trained):
    bundle = tmp_path / "bundle"
    shutil.copytree(trained.bundle, bundle)
    data = tmp_path / "data.arff"
    shutil.copyfile(DATA, data)
    metrics = json.loads((bundle / "metrics.json").read_text())
    metrics["materials"]["dataset"]["path"] = str(data)  # as if trained from this copy
    (bundle / "metrics.json").write_text(json.dumps(metrics))
    build_statement(bundle)  # consistent: fine
    data.write_bytes(DATA.read_bytes() + b"\n")  # the dataset moves on after training
    with pytest.raises(ProvenanceError, match="dataset changed since training"):
        build_statement(bundle)


def test_statement_refuses_an_incomplete_bundle(tmp_path, trained):
    shutil.copytree(trained.bundle, tmp_path / "b")
    (tmp_path / "b" / "gates.json").unlink()
    with pytest.raises(ProvenanceError, match="incomplete"):
        build_statement(tmp_path / "b")


def test_git_source_outside_a_repository(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    assert git_source(tmp_path)["commit"] is None


# --- quality gates ---------------------------------------------------------------------------


def _names(results, passed):
    return {r.name for r in results if r.passed is passed}


def test_gates_pass_on_the_trained_model(trained, trusted_cfg):
    results = run_gates(trained.metrics, trusted_cfg.gates, trained.metrics)
    assert all(r.passed for r in results), results


def test_metric_floor_blocks_a_weak_model(trained, trusted_cfg):
    weak = copy.deepcopy(trained.metrics)
    weak["test"]["metrics"]["recall"] = 0.80
    assert "min_recall" in _names(run_gates(weak, trusted_cfg.gates, None), False)


def test_regression_against_the_baseline_is_blocked(trained, trusted_cfg):
    baseline = copy.deepcopy(trained.metrics)
    baseline["test"]["metrics"]["f1"] += 0.02  # the approved model was 2 points better
    results = run_gates(trained.metrics, trusted_cfg.gates, baseline)
    assert _names(results, False) == {"regression_f1"}


def test_export_parity_is_a_gate(trained, trusted_cfg):
    drifted = copy.deepcopy(trained.metrics)
    drifted["test"]["export_max_abs_diff"] = 1e-3
    assert "export_parity" in _names(run_gates(drifted, trusted_cfg.gates, None), False)


def _reproducible(fresh, committed, cfg):
    results = run_gates(fresh, cfg.gates, committed, expect_reproducible=True)
    return next(r for r in results if r.name == "reproducible")


def test_reproducibility_gate(trained, trusted_cfg):
    assert _reproducible(trained.metrics, trained.metrics, trusted_cfg).passed
    no_baseline = run_gates(trained.metrics, trusted_cfg.gates, None, expect_reproducible=True)
    assert "reproducible" in _names(no_baseline, False)


@pytest.mark.parametrize(
    "path",
    [
        ("test", "metrics", "roc_auc"),
        ("test", "ci95", "f1", 0),
        ("test", "confusion", "fp"),
        ("selection", "candidates", 0, "cv", "roc_auc", "mean"),
        ("duplicates_effect", "naive", "accuracy", "mean"),
        ("curves", "selected", "roc", "tpr", 1),
        ("data", "test_rows"),
    ],
)
def test_reproducibility_gate_covers_the_whole_report(trained, trusted_cfg, path):
    """A stale or hand-edited interval, count or figure input fails, not just a metric."""
    edited = copy.deepcopy(trained.metrics)
    *parents, last = path
    node = edited
    for key in parents:
        node = node[key]
    node[last] += 1e-4
    result = _reproducible(edited, trained.metrics, trusted_cfg)
    assert not result.passed
    where = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path).lstrip(".")
    assert f"at {where} " in result.detail


def test_reproducibility_gate_ignores_provenance_but_not_the_data(trained, trusted_cfg):
    elsewhere = copy.deepcopy(trained.metrics)  # another machine, commit or OS
    elsewhere["materials"]["source"]["commit"] = "0" * 40
    elsewhere["environment"]["platform"] = "Linux-x86_64"
    elsewhere["model"]["sha256"] = "f" * 64
    assert _reproducible(elsewhere, trained.metrics, trusted_cfg).passed
    other_data = copy.deepcopy(trained.metrics)
    other_data["materials"]["dataset"]["sha256"] = "0" * 64
    assert not _reproducible(other_data, trained.metrics, trusted_cfg).passed
    missing = copy.deepcopy(trained.metrics)
    del missing["duplicates_effect"]["grouped"]
    assert not _reproducible(missing, trained.metrics, trusted_cfg).passed
