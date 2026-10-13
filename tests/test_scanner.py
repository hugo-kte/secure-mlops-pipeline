"""Pickle scanner tests.

Every pickle built here is harmless: it references `print`, `len` or a module that does not
exist. Nothing is ever unpickled. Known code-execution primitives are only exercised as
strings through `classify`, never embedded in a pickle.
"""

import collections
import gzip
import io
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mlseal import scanner
from mlseal.cli import app
from mlseal.scanner import Severity, classify, scan_bytes, scan_path


class CallsPrint:
    def __reduce__(self):
        return (print, ("hello",))


class CallsLen:
    def __reduce__(self):
        return (len, ([1, 2, 3],))


def _severities(report) -> set[Severity]:
    return {f.severity for f in report.findings}


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_plain_data_is_safe(protocol):
    report = scan_bytes(pickle.dumps({"a": [1, 2.5, "x"], "b": (None, True)}, protocol))
    assert report.safe
    assert report.globals == []


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_allowlisted_globals_are_safe(protocol):
    report = scan_bytes(pickle.dumps(collections.OrderedDict(a=1), protocol))
    assert report.safe, report.findings
    assert "collections.OrderedDict" in report.globals


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_numpy_arrays_are_safe(protocol):
    report = scan_bytes(pickle.dumps(np.arange(6, dtype=np.float32).reshape(2, 3), protocol))
    assert report.safe, report.findings


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("obj", [CallsPrint(), CallsLen()])
def test_non_allowlisted_callable_is_flagged(obj, protocol):
    report = scan_bytes(pickle.dumps(obj, protocol))  # GLOBAL (0-3) or STACK_GLOBAL (4-5)
    assert not report.safe
    assert _severities(report) == {Severity.HIGH}
    assert any(g.endswith((".print", ".len")) for g in report.globals)


def test_memoized_strings_are_resolved():
    raw = (
        b"\x80\x04"
        b"\x8c\x08builtins\x94"  # SHORT_BINUNICODE + MEMOIZE (memo 0)
        b"\x8c\x05print\x94"  # memo 1
        b"00"  # POP POP: the names are only reachable through the memo now
        b"h\x00h\x01"  # BINGET 0, BINGET 1
        b"\x93)R."  # STACK_GLOBAL, EMPTY_TUPLE, REDUCE, STOP
    )
    report = scan_bytes(raw)
    assert report.globals == ["builtins.print"]
    assert not report.safe


def test_unresolvable_global_fails_closed():
    raw = b"\x80\x04K\x01K\x02\x93."  # STACK_GLOBAL built from two integers
    report = scan_bytes(raw)
    assert Severity.CRITICAL in _severities(report)


def test_scanner_never_imports_what_it_reads():
    name = "mlseal_nonexistent_module_for_scanner_test"
    raw = b"c" + name.encode() + b"\nfunc\n(tR."  # protocol 0: GLOBAL, MARK, TUPLE, REDUCE
    report = scan_bytes(raw, "model.pkl")
    assert f"{name}.func" in report.globals
    assert not report.safe
    assert name not in sys.modules


def test_scikit_learn_pickle_is_refused_by_default():
    from sklearn.linear_model import LogisticRegression

    est = LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    report = scan_bytes(pickle.dumps(est))
    assert not report.safe
    assert any("sklearn.linear_model" in g for g in report.globals)


@pytest.mark.parametrize(
    ("module", "name", "expected"),
    [
        ("os", "system", Severity.CRITICAL),
        ("posix", "system", Severity.CRITICAL),
        ("nt", "system", Severity.CRITICAL),
        ("subprocess", "Popen", Severity.CRITICAL),
        ("builtins", "eval", Severity.CRITICAL),
        ("builtins", "exec", Severity.CRITICAL),
        ("builtins", "getattr", Severity.CRITICAL),
        ("__builtin__", "__import__", Severity.CRITICAL),
        ("functools", "partial", Severity.CRITICAL),
        ("operator", "attrgetter", Severity.CRITICAL),
        ("importlib.machinery", "SourceFileLoader", Severity.CRITICAL),
        ("torch", "load", Severity.CRITICAL),
        ("builtins", "print", Severity.HIGH),
        ("sklearn.ensemble._gb", "GradientBoostingClassifier", Severity.HIGH),
        ("collections", "OrderedDict", Severity.INFO),
        ("numpy._core.multiarray", "_reconstruct", Severity.INFO),
    ],
)
def test_classification(module, name, expected):
    assert classify(module, name) is expected


def test_truncated_pickle_fails_closed():
    report = scan_bytes(pickle.dumps(collections.OrderedDict(a=1), 4)[:-6], "model.pkl")
    assert not report.safe


def test_text_files_are_not_mistaken_for_pickles():
    texts = (
        b"Introduction\nThis is a README.\n",
        b'{"format": "x"}',
        b"# Title\n",
        b"Python 3.12\nnotes\n",  # starts with an opcode that records a persistent id
        b"Makefile targets\n",
        b"(see below)\n",
        b"plain words\n",
    )
    for text in texts:
        report = scan_bytes(text, "notes.txt")
        assert report.kind == "not-pickle", text
        assert report.safe


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_pickles_without_a_protocol_header_are_scanned(protocol):
    """`pickle.loads` does not require the PROTO opcode; neither a missing header nor an
    innocent file name may turn a pickle into "not a pickle"."""
    raw = pickle.dumps(CallsPrint(), protocol)
    assert raw[:1] == b"\x80"
    report = scan_bytes(raw[2:], "weights.dat")  # PROTO dropped; protocol 4+ opens with FRAME
    assert report.kind == "pickle"
    assert any(g.endswith(".print") for g in report.globals)  # __builtin__ in protocol 2-3
    assert not report.safe


def test_pickles_inside_zip_members_without_a_pickle_name_are_scanned():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data/0", pickle.dumps(CallsLen(), 4)[2:])  # header-less
        zf.writestr("archive/byteorder", b"little")
    report = scan_bytes(buf.getvalue(), "checkpoint.pt")
    assert not report.safe
    assert report.findings[0].member == "archive/data/0"


def test_pickles_inside_zip_archives_are_scanned():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps(CallsPrint(), 2))
        zf.writestr("archive/version", b"3\n")
    report = scan_bytes(buf.getvalue(), "checkpoint.pt")
    assert report.kind == "zip"
    assert not report.safe
    assert report.findings[0].member == "archive/data.pkl"


def test_compressed_pickles_are_scanned():
    report = scan_bytes(gzip.compress(pickle.dumps(CallsLen(), 4)), "model.joblib")
    assert report.kind == "gzip+pickle"
    assert not report.safe


def test_decompression_bomb_is_refused(monkeypatch):
    monkeypatch.setattr(scanner, "MAX_SCAN_BYTES", 1000)
    report = scan_bytes(gzip.compress(b"\x00" * 100_000), "model.pkl")
    assert report.kind == "error"
    assert not report.safe


def test_scan_path_refuses_symlinks_instead_of_following_them(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.pkl").write_bytes(pickle.dumps(CallsPrint()))
    scanned = tmp_path / "scanned"
    (scanned / "sub").mkdir(parents=True)
    (scanned / "model.json").write_text("{}")
    (scanned / "sub" / "ok.pkl").write_bytes(pickle.dumps(collections.OrderedDict(a=1)))
    (scanned / "link.pkl").symlink_to(outside / "model.pkl")  # a loader would follow it
    reports = {Path(r.path).name: r for r in scan_path(scanned)}
    assert set(reports) == {"model.json", "ok.pkl", "link.pkl"}
    assert reports["ok.pkl"].safe
    assert not reports["link.pkl"].safe
    assert "symbolic link" in reports["link.pkl"].findings[0].message
    assert reports["link.pkl"].globals == []  # refused unread, not scanned through the link


def test_scan_path_refuses_a_missing_path(tmp_path):
    (report,) = scan_path(tmp_path / "typo")
    assert not report.safe


def test_scan_command_exit_codes(tmp_path):
    runner = CliRunner()
    assert runner.invoke(app, ["scan", str(tmp_path)]).exit_code == 1  # nothing scanned
    assert runner.invoke(app, ["scan", str(tmp_path / "typo")]).exit_code == 1
    (tmp_path / "model.json").write_text("{}")
    assert runner.invoke(app, ["scan", str(tmp_path)]).exit_code == 0
    (tmp_path / "ok.pkl").write_bytes(pickle.dumps(collections.OrderedDict(a=1)))
    assert runner.invoke(app, ["scan", str(tmp_path)]).exit_code == 0
    result = runner.invoke(app, ["scan", str(tmp_path), "--forbid-pickle"])
    assert result.exit_code == 1  # a release bundle may hold no pickle at all
    (tmp_path / "bad.pkl").write_bytes(pickle.dumps(CallsLen()))
    assert runner.invoke(app, ["scan", str(tmp_path)]).exit_code == 1
