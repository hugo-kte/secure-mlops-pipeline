"""The release flow through the CLI, as CI runs it: train, gate, scan, keygen, sign, verify."""

import json
import pickle
import shutil
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from conftest import ROOT, write_config
from mlseal.cli import app

runner = CliRunner()


def run(*args: str, ok: bool = True):
    result = runner.invoke(app, [str(a) for a in args])
    assert (result.exit_code == 0) is ok, result.output
    return result


def test_full_release_flow(tmp_path):
    # A linear model keeps this end-to-end run fast; floors are lowered to match it.
    config = write_config(
        tmp_path,
        model={"candidates": [{"kind": "logistic_regression", "C": 1.0}]},
        gates={"minimum": {"roc_auc": 0.95, "f1": 0.9, "recall": 0.9}},
    )
    bundle, keys, reports = tmp_path / "bundle", tmp_path / "keys", tmp_path / "reports"
    run("train", "-c", config, "-b", bundle)
    run("gate", "-c", config, "-b", bundle, "--baseline", tmp_path / "none.json")
    run("scan", bundle, "--forbid-pickle")
    run("keygen", "--out-dir", keys)
    run("keygen", "--out-dir", keys, ok=False)  # never overwrites a key
    run("sign", "-b", bundle, "--key", keys / "signing.key")
    out = run("verify", "-b", bundle, "--public-key", keys / "signing.pub").output
    assert "Verified release" in out
    run("promote", "-b", bundle, "--reports", reports)
    assert {p.name for p in reports.iterdir()} == {
        "metrics.json", "MODEL_CARD.md", "data_validation.json", "gates.json", "figures",
    }  # fmt: skip
    assert {p.name for p in (reports / "figures").iterdir()} == {
        "duplicates_effect.png",
        "roc_pr.png",
    }
    # The promoted report is now the baseline: an identical retrain reproduces it exactly.
    run("gate", "-c", config, "-b", bundle, "--baseline", reports / "metrics.json",
        "--expect-reproducible")  # fmt: skip

    (bundle / "model_card.md").write_text("edited after signing")
    run("verify", "-b", bundle, "--public-key", keys / "signing.pub", ok=False)


@pytest.fixture
def unsigned(tmp_path, trained, keys):
    bundle = tmp_path / "bundle"
    shutil.copytree(trained.bundle, bundle)
    return bundle


def test_sign_refuses_a_bundle_that_failed_a_gate(unsigned, keys):
    gates = json.loads((unsigned / "gates.json").read_text())
    (unsigned / "gates.json").write_text(json.dumps({**gates, "passed": False}))
    assert "refusing to sign" in run("sign", "-b", unsigned, "--key", keys.private, ok=False).output


def test_sign_refuses_a_bundle_with_a_pickle(unsigned, keys):
    (unsigned / "extra.pkl").write_bytes(pickle.dumps({"weights": [1, 2]}))
    assert "pickle" in run("sign", "-b", unsigned, "--key", keys.private, ok=False).output


def test_gate_command_fails_below_a_floor(tmp_path, unsigned):
    config = write_config(tmp_path, gates={"minimum": {"roc_auc": 0.9999}})
    out = run("gate", "-c", config, "-b", unsigned, "--baseline", tmp_path / "x", ok=False)
    assert "must not be released" in out.output
    assert json.loads((unsigned / "gates.json").read_text())["passed"] is False


def test_verify_fails_cleanly_without_a_usable_public_key(signed_bundle, tmp_path):
    (tmp_path / "junk.pub").write_text("not a key")
    for key in (tmp_path / "missing.pub", tmp_path / "junk.pub"):
        result = run("verify", "-b", signed_bundle, "--public-key", key, ok=False)
        assert "FAILED" in result.output
        assert not isinstance(result.exception, (OSError, ValueError))  # no traceback


def test_promote_refuses_a_failed_bundle(unsigned, tmp_path):
    (unsigned / "gates.json").write_text(json.dumps({"passed": False, "results": []}))
    run("promote", "-b", unsigned, "--reports", tmp_path / "r", ok=False)


def test_module_entry_point():
    proc = subprocess.run(
        [sys.executable, "-m", "mlseal", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""},
        check=False,
    )
    assert proc.returncode == 0
    for command in ("validate", "train", "gate", "scan", "keygen", "sign", "verify", "serve"):
        assert command in proc.stdout
