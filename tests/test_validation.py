"""The data gate must stop the pipeline on every class of violation it claims to detect."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from conftest import DATA, ROOT, write_config
from mlseal.cli import app
from mlseal.config import load_config
from mlseal.data import parse_arff, sha256_bytes
from mlseal.pipeline import train
from mlseal.validation import (
    DataValidationError,
    load_reference,
    profile,
    psi,
    split_check,
    validate_bytes,
)

RAW = DATA.read_bytes()
HEADER_END = RAW.index(b"@data") + len(b"@data\r\n")


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "pipeline.yaml")


@pytest.fixture(scope="module")
def reference():
    return load_reference(ROOT / "configs" / "reference_profile.json")


def _rows() -> list[list[int]]:
    return parse_arff(RAW).values.tolist()


def _rebuild(rows: list[list[int]], header: bytes | None = None) -> bytes:
    body = "\r\n".join(",".join(str(v) for v in r) for r in rows).encode() + b"\r\n"
    return (header or RAW[:HEADER_END]) + body


def _failed(report) -> set[str]:
    return {c.name for c in report.failures()}


def _repinned(cfg, raw):
    """Same policy but pinned to `raw`, to show the other checks catch it on their own."""
    return cfg.data.policy.model_copy(update={"sha256": sha256_bytes(raw)})


def test_reviewed_file_passes_every_check(cfg, reference):
    report = validate_bytes(RAW, cfg.data.policy, reference)
    assert report.passed, report.failures()
    assert {c.name for c in report.checks} == {
        "integrity", "parse", "schema", "allowed_values", "row_count", "label_balance",
        "duplicates", "label_conflicts", "feature_drift",
    }  # fmt: skip


def test_one_changed_byte_fails_integrity(cfg, reference):
    raw = bytearray(RAW)
    assert raw[-4:] == b"-1\r\n"
    raw[-4] = ord(" ")  # last label -1 -> " 1": still a valid file, one label flipped
    report = validate_bytes(bytes(raw), cfg.data.policy, reference)
    assert "integrity" in _failed(report)
    assert any(c.name == "label_balance" for c in report.checks)  # still fully diagnosed


def test_renamed_attribute_fails_schema(cfg, reference):
    raw = RAW.replace(b"@attribute Favicon", b"@attribute Favicon_v2")
    report = validate_bytes(raw, _repinned(cfg, raw), reference)
    assert _failed(report) == {"schema"}


def test_out_of_domain_value_fails(cfg, reference):
    rows = _rows()
    rows[10][18] = -1  # Redirect only allows {0, 1}
    raw = _rebuild(rows)
    report = validate_bytes(raw, _repinned(cfg, raw), reference)
    assert _failed(report) == {"allowed_values"}


def test_mass_label_flipping_fails_balance(cfg, reference):
    rows = _rows()
    for r in rows[:3000]:
        r[-1] = -1  # relabel 3,000 rows as phishing
    raw = _rebuild(rows)
    assert "label_balance" in _failed(validate_bytes(raw, _repinned(cfg, raw), reference))


def test_flipped_duplicates_fail_conflict_check(cfg, reference):
    rows = _rows()
    first_label: dict[tuple, int] = {}
    flipped: set[tuple] = set()
    for r in rows:  # flip the label of one later copy of 100 duplicated vectors
        key = tuple(r[:-1])
        if key not in first_label:
            first_label[key] = r[-1]
        elif len(flipped) < 100 and key not in flipped and r[-1] == first_label[key]:
            r[-1] = -r[-1]
            flipped.add(key)
    raw = _rebuild(rows)
    report = validate_bytes(raw, _repinned(cfg, raw), reference)
    assert "label_conflicts" in _failed(report)
    assert "label_balance" not in _failed(report)  # a subtle attack the balance check misses


def test_shifted_feature_fails_drift(cfg, reference):
    rows = _rows()
    for r in rows[:4000]:
        r[7] = -1  # SSLfinal_State shifted on 4,000 rows
    raw = _rebuild(rows)
    assert "feature_drift" in _failed(validate_bytes(raw, _repinned(cfg, raw), reference))


def test_truncated_file_fails_row_count(cfg, reference):
    raw = _rebuild(_rows()[:5000])
    assert "row_count" in _failed(validate_bytes(raw, _repinned(cfg, raw), reference))


def test_unparseable_file_fails_closed(cfg, reference):
    report = validate_bytes(b"not an arff file", cfg.data.policy, reference)
    assert not report.passed
    assert {"integrity", "parse"} <= _failed(report)


def test_psi_is_zero_for_identical_and_grows_with_shift():
    a = {"-1": 0.5, "1": 0.5}
    assert psi(a, a) == 0.0
    assert psi(a, {"-1": 0.4, "1": 0.6}) < psi(a, {"-1": 0.1, "1": 0.9})


def test_reference_profile_matches_reviewed_data(reference):
    assert profile(parse_arff(RAW).X) == reference


def test_split_check_detects_overlap():
    X = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    assert split_check(X[:2], X[2:]).passed
    assert not split_check(X[:2], X[1:]).passed


def test_pipeline_refuses_to_train_on_tampered_data(tmp_path):
    rows = _rows()
    rows[0][-1] = -rows[0][-1]  # a single flipped label
    data = tmp_path / "data.arff"
    data.write_bytes(_rebuild(rows))
    config_path = write_config(tmp_path, data_path=data)
    bundle = tmp_path / "bundle"
    with pytest.raises(DataValidationError, match="integrity"):
        train(load_config(config_path), bundle, config_path=config_path, log=lambda _: None)
    report = json.loads((bundle / "data_validation.json").read_text())
    assert report["passed"] is False
    assert not (bundle / "model.json").exists()


def test_validate_command_exits_non_zero_on_violation(tmp_path):
    data = tmp_path / "data.arff"
    data.write_bytes(RAW.replace(b"@attribute port", b"@attribute ports"))
    config_path = write_config(tmp_path, data_path=data)
    result = CliRunner().invoke(
        app, ["validate", "-c", str(config_path), "--out", str(tmp_path / "r.json")]
    )
    assert result.exit_code == 1
    assert "FAILED" in result.output
