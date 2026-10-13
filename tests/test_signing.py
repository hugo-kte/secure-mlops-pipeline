"""Signing and verification: anything but an intact bundle signed by a trusted key is refused."""

import base64
import json
import os
import re
import stat

import pytest

from conftest import Keys
from mlseal.signing import (
    ENVELOPE_NAME,
    VerificationError,
    generate_key,
    key_id,
    load_private_key,
    load_public_key,
    pae,
    save_keypair,
    sign_bundle,
    sign_envelope,
    verify_bundle,
    verify_envelope,
)


@pytest.fixture
def trusted(keys: Keys):
    return [load_public_key(keys.public.read_bytes())]


def test_intact_bundle_verifies(signed_bundle, trusted):
    verified = verify_bundle(signed_bundle, trusted)
    assert set(verified.files) == {
        "model.json", "metrics.json", "model_card.md", "data_validation.json", "gates.json",
    }  # fmt: skip
    assert verified.keyid == key_id(trusted[0])


def test_private_key_is_owner_only_and_never_overwritten(tmp_path):
    private, public = save_keypair(generate_key(), tmp_path / "keys")
    assert stat.S_IMODE(os.stat(private).st_mode) == 0o600
    assert b"PRIVATE KEY" in private.read_bytes()
    assert b"PRIVATE" not in public.read_bytes()
    with pytest.raises(FileExistsError):
        save_keypair(generate_key(), tmp_path / "keys")


def test_encrypted_private_key_needs_its_passphrase(tmp_path):
    private, _ = save_keypair(generate_key(), tmp_path, passphrase=b"correct horse")
    assert b"ENCRYPTED" in private.read_bytes()
    with pytest.raises(TypeError):
        load_private_key(private)  # no passphrase given
    with pytest.raises(ValueError):
        load_private_key(private, b"wrong")
    load_private_key(private, b"correct horse")


def test_untrusted_key_is_refused(signed_bundle):
    other = generate_key().public_key()
    with pytest.raises(VerificationError, match="no signature from a trusted key"):
        verify_bundle(signed_bundle, [other])


def test_no_trusted_key_configured_is_refused(signed_bundle):
    with pytest.raises(VerificationError, match="no trusted public key"):
        verify_bundle(signed_bundle, [])


def test_unsigned_bundle_is_refused(bundle_copy, trusted):
    (bundle_copy / ENVELOPE_NAME).unlink()
    with pytest.raises(VerificationError, match="unsigned bundle"):
        verify_bundle(bundle_copy, trusted)


@pytest.mark.parametrize(
    "name", ["model.json", "metrics.json", "model_card.md", "data_validation.json", "gates.json"]
)
def test_any_modified_file_is_refused(bundle_copy, trusted, name):
    path = bundle_copy / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(VerificationError, match=re.escape(f"digest mismatch for {name}")):
        verify_bundle(bundle_copy, trusted)


def test_missing_file_is_refused(bundle_copy, trusted):
    (bundle_copy / "model.json").unlink()
    with pytest.raises(VerificationError, match=r"cannot open model\.json"):
        verify_bundle(bundle_copy, trusted)


def test_symlinked_subject_is_refused(bundle_copy, trusted, tmp_path):
    real = tmp_path / "elsewhere.json"
    real.write_bytes((bundle_copy / "model.json").read_bytes())  # identical content
    (bundle_copy / "model.json").unlink()
    (bundle_copy / "model.json").symlink_to(real)
    with pytest.raises(VerificationError, match=r"cannot open model\.json"):
        verify_bundle(bundle_copy, trusted)


@pytest.mark.parametrize("make", [os.mkdir, os.mkfifo], ids=["directory", "fifo"])
def test_subject_that_is_not_a_regular_file_is_refused(bundle_copy, trusted, make):
    (bundle_copy / "gates.json").unlink()
    make(bundle_copy / "gates.json")  # a FIFO must not block verification either
    with pytest.raises(VerificationError, match=r"gates\.json is not a regular file"):
        verify_bundle(bundle_copy, trusted)


@pytest.mark.parametrize("extra", ["model.pkl", ".hidden", "sub/"])
def test_files_not_covered_by_the_manifest_are_refused(bundle_copy, trusted, extra):
    if extra.endswith("/"):
        (bundle_copy / extra).mkdir()
    else:
        (bundle_copy / extra).write_bytes(b"not signed")  # e.g. a planted second model
    with pytest.raises(VerificationError, match="not covered by the signed manifest"):
        verify_bundle(bundle_copy, trusted)


def _rewrite_statement(bundle, keys: Keys, mutate):
    """Re-sign a modified statement with the trusted key: checks beyond the signature."""
    env = json.loads((bundle / ENVELOPE_NAME).read_text())
    statement = json.loads(base64.b64decode(env["payload"]))
    mutate(statement)
    sign_bundle(bundle, statement, load_private_key(keys.private))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s["subject"].__setitem__(0, {**s["subject"][0], "name": "../model.json"}),
         "invalid subject name"),
        (lambda s: s["subject"].append(dict(s["subject"][1])), "invalid subject name"),
        (lambda s: s.update(predicateType="https://example.org/other"), "predicate type"),
        (lambda s: s.update(_type="https://in-toto.io/Statement/v0.1"), "in-toto v1"),
        (lambda s: s.update(subject=[x for x in s["subject"] if x["name"] != "model.json"]),
         "does not cover model.json"),
        (lambda s: s.update(subject=[]), "no subjects"),
        (lambda s: s["subject"][0].update(digest=["sha256"]), "no valid sha256"),
        (lambda s: s["subject"][0].update(digest={"sha256": "é" * 64}), "no valid sha256"),
    ],
)  # fmt: skip
def test_signed_but_invalid_statements_are_refused(bundle_copy, keys, trusted, mutate, message):
    _rewrite_statement(bundle_copy, keys, mutate)
    with pytest.raises(VerificationError, match=message):
        verify_bundle(bundle_copy, trusted)


def test_payload_swap_breaks_the_signature(keys, trusted):
    key = load_private_key(keys.private)
    env = sign_envelope(b'{"a": 1}', key)
    env["payload"] = base64.b64encode(b'{"a": 2}').decode()
    with pytest.raises(VerificationError, match="no signature from a trusted key"):
        verify_envelope(json.dumps(env).encode(), trusted)


def test_payload_type_is_bound_into_the_signature(keys, trusted):
    key = load_private_key(keys.private)
    env = sign_envelope(b"{}", key)
    assert pae("a", b"x") != pae("b", b"x")
    env["payloadType"] = "text/plain"
    with pytest.raises(VerificationError, match="payload type"):
        verify_envelope(json.dumps(env).encode(), trusted)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"payloadType": "application/vnd.in-toto+json", "payload": "e30=", "signatures": []}',
        b'{"payloadType": "application/vnd.in-toto+json", "payload": "***", "signatures": [{}]}',
        b'{"payload": "e30=", "payload": "e30=", "payloadType": "x", "signatures": []}',
        b"not json",
        b'{"payload": "", "payloadType": "x", "signatures": [' + b"1" * 5000 + b"]}",
    ],
)
def test_malformed_envelopes_are_refused(raw, trusted):
    with pytest.raises(VerificationError):
        verify_envelope(raw, trusted)


def test_key_ids_are_hints_only(keys, trusted):
    env = sign_envelope(b"{}", load_private_key(keys.private))
    env["signatures"][0]["keyid"] = "0" * 64  # wrong hint, valid signature
    payload, keyid = verify_envelope(json.dumps(env).encode(), trusted)
    assert payload == b"{}"
    assert keyid == key_id(trusted[0])
