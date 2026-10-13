import io
import zipfile

import numpy as np
import pytest

from conftest import DATA
from mlseal.data import (
    DataError,
    archive_member_digest,
    deduplicate,
    group_ids,
    overlap,
    parse_arff,
    read_arff,
    sha256_bytes,
    sha256_file,
)
from mlseal.schema import FEATURE_NAMES, declared_attributes

PINNED = "3bcf93b1612306537ee2737751f1bac185b2a77886ab376644c3bf0c7f7cb3e6"


def test_committed_dataset_matches_pinned_digest():
    assert sha256_file(DATA) == PINNED


def test_reads_the_real_file():
    ds = read_arff(DATA)
    assert ds.values.shape == (11055, 31)
    declared = [(name, sorted(domain)) for name, domain in ds.attributes]
    assert declared == [(name, sorted(domain)) for name, domain in declared_attributes()]
    assert int(ds.y.sum()) == 4898  # Result == -1 (phishing)
    assert ds.X.dtype == np.int8


def _arff(rows: list[str], header: str | None = None) -> bytes:
    header = header or "@relation t\n@attribute a {-1,1}\n@attribute Result {-1,1}\n@data\n"
    return (header + "\n".join(rows) + "\n").encode()


def test_parser_accepts_crlf_and_comments():
    raw = b"% comment\r\n@relation t\r\n@attribute a {-1,1}\r\n@attribute Result {-1,1}\r\n"
    ds = parse_arff(raw + b"@data\r\n1,-1\r\n-1,1\r\n")
    assert ds.values.tolist() == [[1, -1], [-1, 1]]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_arff(["1,?"]), "non-integer or missing"),
        (_arff(["1,1,1"]), "3 values, expected 2"),
        (_arff(["1,abc"]), "non-integer"),
        (_arff([]), "no data rows"),
        (b"@relation t\n@attribute a {-1,1}\n", "no @data"),
        (b"@relation t\n@attribute a numeric\n@data\n1\n", "only nominal"),
        (b"@attribute a {-1,1}\n@data\n1\n", "unexpected header"),
        ("@relation t\n@attribute é {1}\n@data\n1\n".encode(), "not ASCII"),
        (_arff(["99999,1"]), "out of range"),
    ],
)
def test_parser_rejects_malformed_files(raw, message):
    with pytest.raises(DataError, match=message):
        parse_arff(raw)


def test_deduplication_statistics_on_the_real_file():
    ds = read_arff(DATA)
    Xu, yu, stats = deduplicate(ds.X, ds.y)
    assert stats.rows == 11055
    assert stats.unique_vectors == 5785
    assert stats.duplicate_rows == 5270
    assert stats.conflicting_vectors == 64
    assert stats.ties_dropped == 13
    assert len(Xu) == len(yu) == 5772 == stats.phishing + stats.legitimate
    assert len(np.unique(group_ids(Xu))) == len(Xu)  # really one row per vector


def test_deduplicate_uses_majority_and_drops_ties():
    X = np.array([[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [-1, 1]], dtype=np.int8)
    y = np.array([1, 1, 0, 1, 0, 0], dtype=np.int8)
    Xu, yu, stats = deduplicate(X, y)
    assert Xu.tolist() == [[1, 1], [-1, 1]]  # first appearance order; the tie [0, 0] is gone
    assert yu.tolist() == [1, 0]
    assert (stats.conflicting_vectors, stats.ties_dropped) == (2, 1)


def test_overlap_counts_shared_vectors():
    a = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    b = np.array([[1, 1], [1, 1], [-1, -1]], dtype=np.int8)
    assert overlap(a, b) == 1
    assert overlap(a, -a - 2) == 0


def test_archive_member_digest_reads_without_extracting():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Training Dataset.arff", DATA.read_bytes())
        zf.writestr("other.txt", b"x")
    assert archive_member_digest(buf.getvalue(), "Training Dataset.arff") == PINNED
    assert sha256_bytes(b"x") != PINNED


def test_feature_contract_has_thirty_features():
    assert len(FEATURE_NAMES) == 30 == len(set(FEATURE_NAMES))
