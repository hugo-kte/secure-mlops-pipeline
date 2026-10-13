"""Ed25519 signing of the release manifest (DSSE envelope) and verification before loading.

Key handling
------------
* `mlseal keygen` writes `signing.key` (PKCS#8 PEM, mode 0600, encrypted when
  MLSEAL_KEY_PASSPHRASE is set) and `signing.pub`. It refuses to overwrite an existing key.
* No private key is ever committed: keys live outside the repository (`runtime/keys/` is
  git-ignored, gitleaks scans every push). CI signs with an ephemeral key generated in the
  job and discarded with the runner; its long-lived trust comes from Sigstore keyless
  attestations (GitHub OIDC), so there is no signing secret to steal.
* In production the private key would sit in a KMS/HSM or an environment-protected CI
  secret, and the server would receive only the public key.

Verification is all-or-nothing: a trusted signature over the exact envelope payload, then the
SHA-256 of every subject file, read once into memory so the bytes that were checked are the
bytes that get parsed (no time-of-check/time-of-use gap). A bundle may hold nothing but the
envelope and the files it signs: an unsigned extra file is refused, not ignored.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from mlseal.provenance import PREDICATE_TYPE, STATEMENT_TYPE

PAYLOAD_TYPE = "application/vnd.in-toto+json"
ENVELOPE_NAME = "manifest.dsse.json"
MAX_ENVELOPE_BYTES = 1024 * 1024
MAX_SUBJECT_BYTES = 32 * 1024 * 1024
_SUBJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class VerificationError(RuntimeError):
    """The bundle is unsigned, signed by an untrusted key, or does not match its manifest."""


# --- keys ---------------------------------------------------------------------------------


def generate_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()


def public_pem(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def save_keypair(
    key: Ed25519PrivateKey, out_dir: Path, passphrase: bytes | None = None
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_path, public_path = out_dir / "signing.key", out_dir / "signing.pub"
    if private_path.exists() or public_path.exists():
        raise FileExistsError(f"refusing to overwrite keys in {out_dir}")
    encryption = (
        serialization.BestAvailableEncryption(passphrase)
        if passphrase
        else serialization.NoEncryption()
    )
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption
    )
    # O_EXCL: never follow or clobber an existing file; 0600: owner-only from the first byte.
    fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    public_path.write_bytes(public_pem(key.public_key()))
    return private_path, public_path


def load_private_key(path: Path, passphrase: bytes | None = None) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=passphrase)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("signing key must be Ed25519")
    return key


def load_public_key(data: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(data)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("public key must be Ed25519")
    return key


# --- DSSE ---------------------------------------------------------------------------------


def pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE pre-authentication encoding: binds the payload type into the signature."""
    t = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)


def sign_envelope(payload: bytes, key: Ed25519PrivateKey) -> dict[str, Any]:
    return {
        "payloadType": PAYLOAD_TYPE,
        "payload": base64.b64encode(payload).decode("ascii"),
        "signatures": [
            {
                "keyid": key_id(key.public_key()),
                "sig": base64.b64encode(key.sign(pae(PAYLOAD_TYPE, payload))).decode("ascii"),
            }
        ],
    }


def _strict_json(data: bytes, what: str) -> Any:
    def no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise VerificationError(f"{what}: duplicate keys")
        return dict(pairs)

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=no_dupes)
    except (ValueError, RecursionError) as exc:  # bad UTF-8 or JSON, oversized integer
        raise VerificationError(f"{what}: not valid JSON") from exc


def _b64(value: Any, what: str) -> bytes:
    if not isinstance(value, str):
        raise VerificationError(f"{what}: expected base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VerificationError(f"{what}: invalid base64") from exc


def verify_envelope(data: bytes, trusted: list[Ed25519PublicKey]) -> tuple[bytes, str]:
    """Return (payload, keyid) if a trusted key signed the envelope; raise otherwise."""
    if len(data) > MAX_ENVELOPE_BYTES:
        raise VerificationError("envelope too large")
    env = _strict_json(data, "envelope")
    if not isinstance(env, dict) or set(env) != {"payloadType", "payload", "signatures"}:
        raise VerificationError("envelope: unexpected structure")
    if env["payloadType"] != PAYLOAD_TYPE:
        raise VerificationError("envelope: unexpected payload type")
    payload = _b64(env["payload"], "payload")
    sigs = env["signatures"]
    if not isinstance(sigs, list) or not sigs:
        raise VerificationError("envelope: no signatures")
    message = pae(PAYLOAD_TYPE, payload)
    # Key ids are hints only: every signature is tried against every trusted key.
    for entry in sigs:
        if not isinstance(entry, dict) or "sig" not in entry:
            raise VerificationError("envelope: malformed signature entry")
        signature = _b64(entry["sig"], "signature")
        for key in trusted:
            try:
                key.verify(signature, message)
            except InvalidSignature:
                continue
            return payload, key_id(key)
    raise VerificationError("no signature from a trusted key")


# --- bundles ------------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedBundle:
    statement: dict[str, Any]
    keyid: str
    files: dict[str, bytes]  # subject name -> verified content


def sign_bundle(bundle_dir: Path, statement: dict[str, Any], key: Ed25519PrivateKey) -> Path:
    payload = json.dumps(statement, indent=2, sort_keys=True).encode("utf-8")
    path = bundle_dir / ENVELOPE_NAME
    path.write_text(json.dumps(sign_envelope(payload, key), indent=2) + "\n", encoding="utf-8")
    return path


def _read_regular_file(path: Path, limit: int) -> bytes:
    # O_NOFOLLOW: a subject replaced by a symlink is refused, not followed. O_NONBLOCK: a
    # FIFO cannot stall startup; it, a directory or a device is refused once opened.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VerificationError(f"cannot open {path.name}: {exc.strerror}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise VerificationError(f"{path.name} is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as fh:
            data = fh.read(limit + 1)
    finally:
        os.close(fd)
    if len(data) > limit:
        raise VerificationError(f"{path.name} exceeds {limit} bytes")
    return data


def verify_bundle(bundle_dir: Path, trusted: list[Ed25519PublicKey]) -> VerifiedBundle:
    if not trusted:
        raise VerificationError("no trusted public key configured")
    envelope_path = bundle_dir / ENVELOPE_NAME
    if not envelope_path.is_file():
        raise VerificationError(f"unsigned bundle: {ENVELOPE_NAME} not found")
    payload, keyid = verify_envelope(_read_regular_file(envelope_path, MAX_ENVELOPE_BYTES), trusted)
    statement = _strict_json(payload, "statement")
    if not isinstance(statement, dict) or statement.get("_type") != STATEMENT_TYPE:
        raise VerificationError("statement: not an in-toto v1 statement")
    if statement.get("predicateType") != PREDICATE_TYPE:
        raise VerificationError("statement: unexpected predicate type")
    subjects = statement.get("subject")
    if not isinstance(subjects, list) or not subjects:
        raise VerificationError("statement: no subjects")

    files: dict[str, bytes] = {}
    for subject in subjects:
        subject = subject if isinstance(subject, dict) else {}
        name, digests = subject.get("name"), subject.get("digest")
        digest = digests.get("sha256") if isinstance(digests, dict) else None
        if not isinstance(name, str) or not _SUBJECT_NAME.match(name) or name in files:
            raise VerificationError(f"statement: invalid subject name {name!r}")
        if not isinstance(digest, str) or not _SHA256_HEX.match(digest):
            raise VerificationError(f"statement: no valid sha256 for {name}")
        content = _read_regular_file(bundle_dir / name, MAX_SUBJECT_BYTES)
        if not hmac.compare_digest(hashlib.sha256(content).hexdigest(), digest):
            raise VerificationError(f"digest mismatch for {name}: file was modified")
        files[name] = content
    if "model.json" not in files:
        raise VerificationError("statement does not cover model.json")
    # Nothing unsigned may ride along: whoever consumes a verified bundle must be able to
    # trust every file in it, not only the ones this server happens to load.
    unsigned = sorted(set(os.listdir(bundle_dir)) - set(files) - {ENVELOPE_NAME})
    if unsigned:
        raise VerificationError(f"files not covered by the signed manifest: {unsigned}")
    return VerifiedBundle(statement=statement, keyid=keyid, files=files)
