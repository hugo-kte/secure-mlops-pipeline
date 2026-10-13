"""Static scanner for pickle-based model files. It never unpickles anything.

Loading a pickle runs whatever callables its GLOBAL / STACK_GLOBAL / INST opcodes import,
which is why a downloaded `.pkl`, `.joblib` or PyTorch checkpoint is effectively code. The
scanner walks the opcode stream with `pickletools.genops`, replays just enough of the
pickle machine's stack to resolve every imported `module.name` (including names assembled
from memoized strings), and checks each one against an allowlist.

Verdicts fail closed: a global that is not allowlisted, a name the scanner cannot resolve, or
a stream it cannot parse all make the file unsafe. Supported containers: raw pickles,
gzip/bz2/xz/zlib-compressed pickles and zip archives (PyTorch's layout); members are size-
capped against decompression bombs.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import pickletools
import zipfile
import zlib
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

MAX_SCAN_BYTES = 512 * 1024 * 1024  # per stream, after decompression
MAX_ARCHIVE_BYTES = 2 * MAX_SCAN_BYTES  # all members of one archive together
MAX_ZIP_MEMBERS = 10_000
# Extensions that are pickles (or pickle containers) in practice: parse errors are findings.
PICKLE_SUFFIXES = frozenset(
    {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".bin", ".sav", ".dill"}
)

# Plain data containers and the numpy reconstruction helpers scikit-learn pickles need.
SAFE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "bytearray"),
        ("builtins", "complex"),
        ("builtins", "slice"),
        ("_codecs", "encode"),  # bytes in protocol-2 pickles
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("datetime", "date"),
        ("datetime", "datetime"),
        ("datetime", "timedelta"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),  # arrays pickled with protocol 5
        ("numpy._core.numeric", "_frombuffer"),
    }
)

# Modules and names known to execute code, touch the OS or chain into other gadgets.
DANGEROUS_MODULES: frozenset[str] = frozenset(
    {
        "os", "posix", "nt", "subprocess", "sys", "socket", "shutil", "importlib", "runpy",
        "pty", "code", "codeop", "ctypes", "multiprocessing", "asyncio", "pickle", "_pickle",
        "marshal", "dill", "cloudpickle", "types", "webbrowser", "urllib", "http", "requests",
        "httpx", "signal", "io", "_io", "functools", "operator", "torch.hub",
    }
)  # fmt: skip
# Functions that deserialise again (a nested pickle the scanner cannot see).
DANGEROUS_NAMES: frozenset[tuple[str, str]] = frozenset(
    {("joblib", "load"), ("torch", "load"), ("numpy", "load"), ("pandas", "read_pickle")}
)
DANGEROUS_BUILTINS: frozenset[str] = frozenset(
    {
        "eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr",
        "globals", "locals", "vars", "input", "breakpoint", "help", "exit", "quit",
    }
)  # fmt: skip

_STRING_OPS = frozenset(
    {
        "STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "BINUNICODE", "SHORT_BINUNICODE",
        "BINUNICODE8",
    }
)  # fmt: skip
_GET_OPS = frozenset({"GET", "BINGET", "LONG_BINGET"})
_PUT_OPS = frozenset({"PUT", "BINPUT", "LONG_BINPUT"})


class Severity(IntEnum):
    INFO = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True)
class Finding:
    severity: Severity
    message: str
    member: str | None = None


@dataclass
class ScanReport:
    path: str
    kind: str  # pickle | zip | compressed+pickle | not-pickle | error
    globals: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return all(f.severity < Severity.HIGH for f in self.findings)

    @property
    def is_pickle(self) -> bool:
        return self.kind not in ("not-pickle",)


def classify(module: str, name: str) -> Severity:
    """Severity of importing `module.name` while unpickling."""
    if (module, name) in SAFE_GLOBALS:
        return Severity.INFO
    root = module.split(".", 1)[0]
    if module in DANGEROUS_MODULES or root in DANGEROUS_MODULES:
        return Severity.CRITICAL
    if (module, name) in DANGEROUS_NAMES:
        return Severity.CRITICAL
    if module in ("builtins", "__builtin__") and name in DANGEROUS_BUILTINS:
        return Severity.CRITICAL
    return Severity.HIGH


class _Mark:
    pass


_MARK = _Mark()
_UNKNOWN = object()


def _scan_stream(data: bytes, member: str | None, report: ScanReport) -> None:
    """Scan one or more concatenated pickles in `data`."""
    pos = 0
    while pos < len(data):
        stack: list[object] = []
        memo: dict[int, object] = {}
        stream = io.BytesIO(data[pos:])
        try:
            for op, arg, _ in pickletools.genops(stream):
                _step(op, arg, stack, memo, member, report)
        except Exception as exc:  # malformed or truncated stream: fail closed
            report.findings.append(
                Finding(Severity.HIGH, f"unparseable pickle stream: {exc}", member)
            )
            return
        consumed = stream.tell()
        if consumed == 0:
            return
        pos += consumed
        if not data[pos:].strip(b"\x00\r\n\t "):
            return


def _record(module: object, name: object, member: str | None, report: ScanReport) -> None:
    if not isinstance(module, str) or not isinstance(name, str):
        report.findings.append(
            Finding(
                Severity.CRITICAL, "global built from values the scanner cannot resolve", member
            )
        )
        return
    qualified = f"{module}.{name}"
    report.globals.append(qualified)
    severity = classify(module, name)
    if severity is not Severity.INFO:
        label = "dangerous" if severity is Severity.CRITICAL else "non-allowlisted"
        report.findings.append(Finding(severity, f"{label} global {qualified}", member))


def _step(
    op: pickletools.OpcodeInfo,
    arg: object,
    stack: list[object],
    memo: dict[int, object],
    member: str | None,
    report: ScanReport,
) -> None:
    name = op.name
    if name in _PUT_OPS:
        memo[int(arg)] = stack[-1] if stack else _UNKNOWN
        return
    if name == "MEMOIZE":
        memo[len(memo)] = stack[-1] if stack else _UNKNOWN
        return
    if name == "DUP":
        stack.append(stack[-1] if stack else _UNKNOWN)
        return
    if name in ("GLOBAL", "INST"):
        module, _, attr = str(arg).partition(" ")
        _record(module, attr, member, report)
    elif name in ("EXT1", "EXT2", "EXT4"):
        report.findings.append(
            Finding(Severity.CRITICAL, "copyreg extension code (unresolvable global)", member)
        )
    elif name in ("PERSID", "BINPERSID"):
        report.findings.append(
            Finding(Severity.MEDIUM, "persistent id: loader-defined objects", member)
        )

    # Generic stack effect, mirroring pickletools.dis: pop through a mark if the opcode
    # consumes one, then pop the remaining operands and push the results.
    before, after = op.stack_before, op.stack_after
    n_pop = len(before)
    if pickletools.markobject in before or (name == "POP" and stack and stack[-1] is _MARK):
        while stack and stack[-1] is not _MARK:
            stack.pop()
        if not stack:
            raise ValueError(f"{name} without a matching MARK")
        stack.pop()
        n_pop = before.index(pickletools.markobject) if pickletools.markobject in before else 0
    if n_pop > len(stack):
        raise ValueError(f"stack underflow at {name}")
    popped = stack[len(stack) - n_pop :] if n_pop else []
    del stack[len(stack) - n_pop :]

    if name == "STACK_GLOBAL":
        _record(popped[0], popped[1], member, report)
        stack.append(_UNKNOWN)
    elif name in _STRING_OPS:
        stack.append(arg.decode("latin-1") if isinstance(arg, bytes) else arg)
    elif name in _GET_OPS:
        stack.append(memo.get(int(arg), _UNKNOWN))
    else:
        stack.extend(_MARK if item is pickletools.markobject else _UNKNOWN for item in after)


def _read_capped(fh: io.BufferedIOBase, limit: int) -> bytes:
    data = fh.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"decompressed size exceeds {limit} bytes")
    return data


def _decompress(data: bytes) -> tuple[str, bytes] | None:
    if data[:2] == b"\x1f\x8b":
        return "gzip", _read_capped(gzip.GzipFile(fileobj=io.BytesIO(data)), MAX_SCAN_BYTES)
    if data[:3] == b"BZh":
        return "bz2", _read_capped(bz2.BZ2File(io.BytesIO(data)), MAX_SCAN_BYTES)
    if data[:6] == b"\xfd7zXZ\x00":
        return "xz", _read_capped(lzma.LZMAFile(io.BytesIO(data)), MAX_SCAN_BYTES)
    if data[:1] == b"\x78" and data[1:2] in (b"\x01", b"\x5e", b"\x9c", b"\xda"):
        d = zlib.decompressobj()
        out = d.decompress(data, MAX_SCAN_BYTES + 1)
        if len(out) > MAX_SCAN_BYTES:
            raise ValueError(f"decompressed size exceeds {MAX_SCAN_BYTES} bytes")
        return "zlib", out
    return None


# Protocols 0-1 have no header; their streams start with one of these opcodes.
_PROTO0_START = frozenset(b"(]}cIJKLNSVFGTUX")


def _scan_members(data: bytes, report: ScanReport) -> None:
    budget = MAX_ARCHIVE_BYTES
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.infolist()
        if len(members) > MAX_ZIP_MEMBERS:
            raise ValueError("too many archive members")
        for info in members:
            if info.is_dir():
                continue
            with zf.open(info) as fh:
                content = _read_capped(fh, min(MAX_SCAN_BYTES, budget))
            budget -= len(content)
            name = info.filename
            if Path(name).suffix.lower() in PICKLE_SUFFIXES or content[:1] == b"\x80":
                _scan_stream(content, name, report)


def scan_bytes(data: bytes, path: str = "<memory>") -> ScanReport:
    """Scan one file's content. `path` is used for reporting and for its suffix."""
    report = ScanReport(path=path, kind="pickle")
    strict = Path(path).suffix.lower() in PICKLE_SUFFIXES
    try:
        if data[:4] == b"PK\x03\x04":
            report.kind = "zip"
            _scan_members(data, report)
            return report
        unpacked = _decompress(data)
        if unpacked is not None:
            report.kind = f"{unpacked[0]}+pickle"
            data, strict = unpacked[1], True
        if data[:1] == b"\x80" or strict:
            _scan_stream(data, None, report)
        elif data[:1] and data[0] in _PROTO0_START:
            # Could be a protocol-0 pickle or just text. Keep it only if it parses as a
            # pickle or reveals a global before failing.
            _scan_stream(data, None, report)
            if not report.globals and not report.safe:
                report.kind, report.findings = "not-pickle", []
        else:
            report.kind = "not-pickle"
    except Exception as exc:  # corrupt archive, decompression bomb, ...: fail closed
        report.kind = "error"
        report.findings.append(Finding(Severity.HIGH, f"cannot scan: {exc}"))
    return report


def scan_path(path: Path) -> list[ScanReport]:
    """Scan a file, or every file under a directory (symlinks are not followed)."""
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink())
    if path.is_file():
        files = [path]
    reports = []
    for file in files:
        if file.stat().st_size > MAX_SCAN_BYTES:
            report = ScanReport(path=str(file), kind="error")
            report.findings.append(Finding(Severity.HIGH, "file too large to scan"))
            reports.append(report)
            continue
        reports.append(scan_bytes(file.read_bytes(), str(file)))
    return reports
