"""Hardened serving: strict schema, limits, quiet errors and logs, fail-closed startup."""

import asyncio
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import DATA
from mlseal import modelfile
from mlseal.data import read_arff
from mlseal.schema import FEATURE_NAMES
from mlseal.serving import Settings, create_app
from mlseal.serving.logs import JsonFormatter
from mlseal.serving.middleware import AccessLog, RateLimit, client_key
from mlseal.signing import VerificationError, generate_key, save_keypair

ROWS = read_arff(DATA).X[:40]
CANARY = "c4n4ry-9f2e-do-not-log"


def instance(i: int = 0) -> dict[str, int]:
    return {name: int(v) for name, v in zip(FEATURE_NAMES, ROWS[i], strict=True)}


def post(client, body, **kwargs):
    return client.post("/v1/predict", json=body, **kwargs)


@pytest.fixture
def settings(signed_bundle, keys):
    return Settings(bundle=signed_bundle, public_keys=(keys.public,), rate_limit_per_minute=0)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_and_readiness(client, signed_bundle):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    model = json.loads((signed_bundle / "metrics.json").read_text())["model"]
    assert ready.json()["model_sha256"] == model["sha256"]


def test_predictions_come_from_the_verified_model(client, signed_bundle):
    body = {"instances": [instance(i) for i in range(5)]}
    response = post(client, body)
    assert response.status_code == 200
    model = modelfile.loads((signed_bundle / "model.json").read_bytes())
    expected = model.predict_proba(ROWS[:5].astype(float))
    got = [p["phishing_probability"] for p in response.json()["predictions"]]
    assert np.allclose(got, np.round(expected, 3))
    labels = [p["label"] for p in response.json()["predictions"]]
    assert labels == ["phishing" if p >= model.threshold else "legitimate" for p in expected]


def test_security_headers(client):
    response = post(client, {"instances": [instance()]})
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert len(response.headers["x-request-id"]) == 32


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_interactive_docs_are_disabled(client, path):
    assert client.get(path).status_code == 404


def _with(**changes):
    inst = instance()
    inst.update(changes)
    return {"instances": [inst]}


@pytest.mark.parametrize(
    "body",
    [
        _with(evil=1),  # unknown field
        _with(having_IP_Address=0),  # not in {-1, 1}
        _with(Redirect=-1),  # not in {0, 1}
        _with(URL_Length=2),
        _with(URL_Length="1"),  # strings are not coerced
        _with(URL_Length=1.0),  # neither are floats
        _with(URL_Length=True),  # nor booleans (True == 1 in Python)
        _with(URL_Length=None),
        {"instances": [{k: v for k, v in instance().items() if k != "SFH"}]},  # missing
        {"instances": []},
        {"instances": [instance()] * 33},  # above the batch limit
        {"instances": [instance()], "debug": True},
        [instance()],
    ],
)
def test_strict_schema_rejects_bad_requests(client, body):
    assert post(client, body).status_code == 422


def test_malformed_json_is_rejected(client):
    response = client.post(
        "/v1/predict", content=b'{"instances": [', headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_errors_never_echo_the_input(client):
    response = post(client, _with(URL_Length=CANARY, **{CANARY[:8]: CANARY}))
    assert response.status_code == 422
    assert CANARY not in response.text
    assert all(set(err) == {"loc", "msg", "type"} for err in response.json()["detail"])


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", None])
def test_only_json_is_accepted(client, content_type):
    headers = {"content-type": content_type} if content_type else {}
    body = json.dumps({"instances": [instance()]}).encode()
    response = client.post("/v1/predict", content=body, headers=headers)
    assert response.status_code == 415


def test_oversized_body_is_rejected_by_declared_length(client):
    body = {"instances": [instance()], "pad": "x" * 40_000}
    assert post(client, body).status_code == 413


def test_oversized_body_is_rejected_without_content_length(client):
    def chunks():
        yield b'{"instances": ['
        for _ in range(400):
            yield json.dumps(instance()).encode() + b","
        yield b"]}"

    response = client.post(
        "/v1/predict", content=chunks(), headers={"content-type": "application/json"}
    )
    assert response.status_code == 413


def test_invalid_content_length_is_rejected(client):
    response = client.post(
        "/v1/predict",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": "abc"},
    )
    assert response.status_code in (400, 413)


def test_rate_limit(signed_bundle, keys):
    settings = Settings(bundle=signed_bundle, public_keys=(keys.public,), rate_limit_per_minute=3)
    with TestClient(create_app(settings)) as client:
        codes = [post(client, {"instances": [instance()]}).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200]
        assert codes[3:] == [429, 429]
        assert client.get("/healthz").status_code == 200  # probes are never throttled


def test_rejections_carry_the_security_headers_and_are_logged(signed_bundle, keys, caplog):
    """The size and rate limits answer before the app: their 413 and 429 must still be
    hardened and logged (throttled clients are the scraping signal the log exists for)."""
    caplog.set_level(logging.INFO)
    settings = Settings(bundle=signed_bundle, public_keys=(keys.public,), rate_limit_per_minute=1)
    with TestClient(create_app(settings)) as client:
        too_big = post(client, {"instances": [instance()], "pad": "x" * 40_000})
        allowed = post(client, {"instances": [instance()]})
        throttled = post(client, {"instances": [instance()]})
    assert [r.status_code for r in (too_big, allowed, throttled)] == [413, 200, 429]
    for response in (too_big, throttled):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"
        assert len(response.headers["x-request-id"]) == 32
    access = [r for r in caplog.records if r.name == "mlseal.access"]
    assert [r.status for r in access] == [413, 200, 429]
    assert [r.request_id for r in access] == [
        r.headers["x-request-id"] for r in (too_big, allowed, throttled)
    ]


def test_access_log_status_when_the_app_fails_or_the_client_leaves(caplog):
    async def fails(scope, receive, send):
        raise RuntimeError("boom")

    async def client_left(scope, receive, send):
        return None  # e.g. BodySizeLimit after an http.disconnect: no response at all

    async def run(app):
        scope = {"type": "http", "method": "POST", "path": "/v1/predict", "headers": []}
        await AccessLog(app, {}, logging.getLogger("mlseal.access"))(scope, None, None)

    caplog.set_level(logging.INFO)
    with pytest.raises(RuntimeError):
        asyncio.run(run(fails))
    asyncio.run(run(client_left))
    assert [r.status for r in caplog.records if r.name == "mlseal.access"] == [500, None]


@pytest.mark.parametrize(
    ("host", "key"),
    [
        ("203.0.113.7", "203.0.113.7"),  # IPv4: one bucket per address, unchanged
        ("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),
        ("2001:DB8:1:2:ffff:ffff:ffff:ffff", "2001:db8:1:2::/64"),  # same /64, same bucket
        ("::ffff:203.0.113.7", "203.0.113.7"),  # IPv4-mapped counts as IPv4
        ("fe80::1%eth0", "fe80::/64"),
        ("testclient", "testclient"),  # not an IP address: used as given
    ],
)
def test_rate_limit_key(host, key):
    assert client_key(host) == key


def test_ipv6_clients_share_a_bucket_per_64_prefix():
    """Rotating through a /64 must not mint fresh quotas; other prefixes and IPv4 peers keep
    their own buckets."""
    served, statuses = [], []

    async def app(scope, receive, send):
        served.append(scope["client"][0])

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    limiter = RateLimit(app, per_minute=1)
    hosts = ["2001:db8:1:2::1", "2001:db8:1:2:dead:beef::9", "2001:db8:1:3::1", "203.0.113.7"]
    for host in [*hosts, "203.0.113.8"]:
        scope = {"type": "http", "path": "/v1/predict", "client": (host, 50000)}
        asyncio.run(limiter(scope, None, send))
    assert served == ["2001:db8:1:2::1", "2001:db8:1:3::1", "203.0.113.7", "203.0.113.8"]
    assert statuses == [429]  # the second address of the first /64


def test_token_bucket_refills():
    limiter = RateLimit(app=None, per_minute=60)
    assert all(limiter.allow("a", now=0.0) for _ in range(60))
    assert not limiter.allow("a", now=0.0)
    assert limiter.allow("a", now=1.0)  # one token per second
    assert limiter.allow("b", now=0.0)  # buckets are per client


def test_rate_limiter_memory_is_bounded():
    limiter = RateLimit(app=None, per_minute=10, max_clients=100)
    for i in range(1000):
        limiter.allow(f"client-{i}", now=0.0)
    assert len(limiter.buckets) == 100


def test_logs_never_contain_request_content(client, caplog):
    caplog.set_level(logging.INFO)
    post(client, {"instances": [instance()]})
    post(client, _with(URL_Length=CANARY))
    access = [r for r in caplog.records if r.name == "mlseal.access"]
    assert len(access) == 2
    assert {r.status for r in access} == {200, 422}
    emitted = "\n".join(JsonFormatter().format(r) for r in caplog.records)  # what hits stdout
    assert CANARY not in emitted
    assert not any(f'"{name}"' in emitted for name in FEATURE_NAMES)
    assert json.loads(JsonFormatter().format(access[0]))["instances"] == 1


def test_model_metadata(client):
    meta = client.get("/v1/model").json()
    assert meta["features"] == list(FEATURE_NAMES)
    assert set(meta["evaluation"]["metrics"]) >= {"roc_auc", "f1"}
    assert len(meta["signed_by"]) == 64


# --- fail-closed startup ---------------------------------------------------------------------


def _start(bundle, public_keys):
    with TestClient(create_app(Settings(bundle=bundle, public_keys=public_keys))):
        pass


def test_refuses_to_start_with_a_tampered_model(bundle_copy, keys):
    model = bundle_copy / "model.json"
    model.write_bytes(model.read_bytes().replace(b'"learning_rate":0.2', b'"learning_rate":0.9'))
    with pytest.raises(VerificationError, match=r"digest mismatch for model\.json"):
        _start(bundle_copy, (keys.public,))


def test_refuses_to_start_when_the_signed_policy_failed(bundle_copy, keys):
    import base64

    from mlseal.signing import load_private_key, sign_bundle

    envelope = json.loads((bundle_copy / "manifest.dsse.json").read_text())
    statement = json.loads(base64.b64decode(envelope["payload"]))
    statement["predicate"]["policy"]["quality_gates"] = "failed"
    sign_bundle(bundle_copy, statement, load_private_key(keys.private))  # validly signed
    with pytest.raises(VerificationError, match="release policy not satisfied"):
        _start(bundle_copy, (keys.public,))


def test_refuses_to_start_unsigned(bundle_copy, keys):
    (bundle_copy / "manifest.dsse.json").unlink()
    with pytest.raises(VerificationError, match="unsigned"):
        _start(bundle_copy, (keys.public,))


def test_refuses_to_start_with_an_untrusted_signer(signed_bundle, tmp_path):
    _, other_public = save_keypair(generate_key(), tmp_path)
    with pytest.raises(VerificationError, match="trusted key"):
        _start(signed_bundle, (other_public,))


def test_refuses_to_start_without_a_public_key(signed_bundle, tmp_path):
    with pytest.raises(VerificationError, match="no trusted public key"):
        _start(signed_bundle, ())
    with pytest.raises(FileNotFoundError):
        _start(signed_bundle, (tmp_path / "missing.pub",))


def test_settings_from_environment(tmp_path):
    s = Settings.from_env(
        {
            "MLSEAL_BUNDLE": str(tmp_path),
            "MLSEAL_PUBLIC_KEY": f"{tmp_path}/a.pub:{tmp_path}/b.pub",
            "MLSEAL_MAX_BATCH": "8",
            "MLSEAL_RATE_LIMIT_PER_MINUTE": "0",
        }
    )
    assert s.bundle == tmp_path
    assert len(s.public_keys) == 2
    assert (s.max_batch, s.rate_limit_per_minute, s.enable_docs) == (8, 0, False)
    with pytest.raises(ValueError):
        Settings.from_env({"MLSEAL_MAX_BATCH": "0"})
