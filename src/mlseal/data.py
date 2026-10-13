"""Strict reader for the UCI Phishing Websites ARFF file, hashing and de-duplication.

The reader only accepts what this dataset needs (nominal integer attributes, dense rows) and
fails loudly on anything else. Whether the declared attributes and values are the *expected*
ones is the data gate's job (`validation.py`), so that every violation is reported at once.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlseal.schema import PHISHING

MAX_DATA_BYTES = 32 * 1024 * 1024


class DataError(ValueError):
    """The file is not a well-formed dense, nominal ARFF file."""


@dataclass(frozen=True)
class Dataset:
    attributes: tuple[tuple[str, tuple[int, ...]], ...]  # declared (name, domain), label last
    values: np.ndarray  # int16 (n_rows, n_attributes), as written in the file

    @property
    def X(self) -> np.ndarray:  # noqa: N802 - X/y naming is idiomatic in ML code
        return self.values[:, :-1].astype(np.int8)

    @property
    def y(self) -> np.ndarray:
        """1 = phishing, 0 = legitimate (the model's positive class is phishing)."""
        return (self.values[:, -1] == PHISHING).astype(np.int8)

    def __len__(self) -> int:
        return int(self.values.shape[0])


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _parse_domain(spec: str, lineno: int) -> tuple[int, ...]:
    spec = spec.strip()
    if not (spec.startswith("{") and spec.endswith("}")):
        raise DataError(f"line {lineno}: only nominal attributes are supported")
    try:
        values = tuple(int(v) for v in spec[1:-1].split(","))
    except ValueError as exc:
        raise DataError(f"line {lineno}: nominal values must be integers") from exc
    if len(set(values)) != len(values):
        raise DataError(f"line {lineno}: duplicate nominal value")
    return values


def parse_arff(raw: bytes) -> Dataset:
    if len(raw) > MAX_DATA_BYTES:
        raise DataError(f"file larger than {MAX_DATA_BYTES} bytes")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise DataError("file is not ASCII") from exc

    attributes: list[tuple[str, tuple[int, ...]]] = []
    rows: list[list[int]] = []
    seen_relation = in_data = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if in_data:
            fields = s.split(",")
            if len(fields) != len(attributes):
                raise DataError(f"line {lineno}: {len(fields)} values, expected {len(attributes)}")
            try:
                rows.append([int(v) for v in fields])
            except ValueError as exc:
                raise DataError(f"line {lineno}: non-integer or missing value") from exc
            continue
        keyword = s.split(maxsplit=1)[0].lower()
        if keyword == "@relation" and not seen_relation and not attributes:
            seen_relation = True
        elif keyword == "@attribute" and seen_relation:
            parts = s.split(maxsplit=2)
            if len(parts) != 3:
                raise DataError(f"line {lineno}: malformed @attribute")
            attributes.append((parts[1], _parse_domain(parts[2], lineno)))
        elif keyword == "@data" and attributes:
            in_data = True
        else:
            raise DataError(f"line {lineno}: unexpected header line")
    if not in_data:
        raise DataError("no @data section")
    if not rows:
        raise DataError("no data rows")
    values = np.asarray(rows, dtype=np.int64)
    if np.abs(values).max() > np.iinfo(np.int16).max:
        raise DataError("value out of range")
    return Dataset(attributes=tuple(attributes), values=values.astype(np.int16))


def read_arff(path: Path) -> Dataset:
    return parse_arff(Path(path).read_bytes())


# --- duplicates -------------------------------------------------------------------------


def group_ids(X: np.ndarray) -> np.ndarray:
    """Integer id per row; identical feature vectors share an id."""
    X = np.ascontiguousarray(X)
    keys = X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
    return np.unique(keys, return_inverse=True)[1].ravel()


@dataclass(frozen=True)
class DedupStats:
    rows: int
    unique_vectors: int
    duplicate_rows: int  # rows whose feature vector already appeared earlier
    conflicting_vectors: int  # vectors seen with both labels
    conflicting_rows: int
    ties_dropped: int  # conflicting vectors with no majority label, removed
    phishing: int  # after de-duplication
    legitimate: int

    @property
    def duplicate_share(self) -> float:
        return self.duplicate_rows / self.rows

    @property
    def conflict_share(self) -> float:
        return self.conflicting_vectors / self.unique_vectors


def deduplicate(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, DedupStats]:
    """One row per distinct feature vector, labelled by majority vote; ties are dropped.

    Rows come out in order of first appearance, so the result is deterministic.
    """
    gid = group_ids(X)
    n_groups = int(gid.max()) + 1
    pos = np.bincount(gid, weights=y, minlength=n_groups)
    size = np.bincount(gid, minlength=n_groups)
    neg = size - pos
    first = np.full(n_groups, len(gid), dtype=np.int64)
    np.minimum.at(first, gid, np.arange(len(gid)))

    conflicting = (pos > 0) & (neg > 0)
    keep = pos != neg
    order = np.argsort(first[keep], kind="stable")
    kept_groups = np.flatnonzero(keep)[order]
    Xu = X[first[kept_groups]]
    yu = (pos[kept_groups] > neg[kept_groups]).astype(np.int8)
    stats = DedupStats(
        rows=len(gid),
        unique_vectors=n_groups,
        duplicate_rows=len(gid) - n_groups,
        conflicting_vectors=int(conflicting.sum()),
        conflicting_rows=int(size[conflicting].sum()),
        ties_dropped=int((~keep).sum()),
        phishing=int(yu.sum()),
        legitimate=int(len(yu) - yu.sum()),
    )
    return Xu, yu, stats


def overlap(X_a: np.ndarray, X_b: np.ndarray) -> int:
    """Number of distinct feature vectors present in both arrays."""
    ids = group_ids(np.vstack([X_a, X_b]))
    return len(np.intersect1d(ids[: len(X_a)], ids[len(X_a) :]))


# --- upstream provenance ------------------------------------------------------------------


def archive_member_digest(archive: bytes, member: str) -> str:
    """SHA-256 of one member of a zip archive, read in memory (nothing is extracted)."""
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        info = zf.getinfo(member)
        if info.file_size > MAX_DATA_BYTES:
            raise DataError(f"{member} is larger than {MAX_DATA_BYTES} bytes")
        with zf.open(info) as fh:
            return sha256_bytes(fh.read(MAX_DATA_BYTES + 1))


def download(url: str, max_bytes: int = 4 * MAX_DATA_BYTES, timeout: float = 60.0) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("only https:// URLs are allowed")
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - https only
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise DataError("download larger than expected")
    return data
