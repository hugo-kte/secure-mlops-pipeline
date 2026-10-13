"""FastAPI application factory.

Startup is fail-closed: the lifespan hook verifies the bundle signature and every digest,
parses the model with the strict loader and checks it against the feature contract. If any
step fails the exception propagates, the server never binds, and the container exits.

(No `from __future__ import annotations` here: FastAPI must resolve the request model, which
is built per app, from the endpoint's real annotation.)
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictInt, create_model

from mlseal import modelfile
from mlseal.schema import FEATURE_NAMES, FEATURES
from mlseal.serving.middleware import BodySizeLimit, RateLimit
from mlseal.serving.settings import Settings
from mlseal.signing import VerificationError, load_public_key, verify_bundle

log = logging.getLogger("mlseal.serving")
access = logging.getLogger("mlseal.access")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Referrer-Policy": "no-referrer",
}


@dataclass(frozen=True)
class LoadedModel:
    model: modelfile.Model
    sha256: str
    keyid: str
    statement: dict[str, Any]


RELEASE_POLICY = {"data_validation": "passed", "quality_gates": "passed", "pickle_scan": "clean"}


def load_verified_model(settings: Settings) -> LoadedModel:
    """Verify, then parse. Raises on an unsigned, untrusted, tampered or non-compliant bundle."""
    trusted = [load_public_key(path.read_bytes()) for path in settings.public_keys]
    bundle = verify_bundle(settings.bundle, trusted)
    # Defence in depth: `mlseal sign` already refuses such bundles, but the server does not
    # rely on every signer having used it.
    policy = bundle.statement.get("predicate", {}).get("policy")
    if policy != RELEASE_POLICY:
        raise VerificationError(f"signed release policy not satisfied: {policy}")
    model = modelfile.loads(bundle.files["model.json"])
    sha = next(
        s["digest"]["sha256"] for s in bundle.statement["subject"] if s["name"] == "model.json"
    )
    return LoadedModel(model=model, sha256=sha, keyid=bundle.keyid, statement=bundle.statement)


def _allowed(values: tuple[int, ...]) -> Any:
    """A strict integer restricted to `values`.

    Not `Literal[...]`: pydantic matches literals by equality, so JSON `true` and `1.0` pass
    as 1 even in strict mode. StrictInt refuses both, then the allowlist applies.
    """
    allowed = frozenset(values)

    def check(value: int) -> int:
        if value not in allowed:
            raise ValueError(f"must be one of {sorted(allowed)}")
        return value

    return Annotated[StrictInt, AfterValidator(check)]


def _request_models(max_batch: int) -> tuple[type[BaseModel], type[BaseModel]]:
    strict = ConfigDict(extra="forbid", strict=True)
    fields: dict[str, Any] = {
        f.name: (
            _allowed(f.values),
            Field(description=f.description, json_schema_extra={"enum": list(f.values)}),
        )
        for f in FEATURES
    }
    features = create_model("Features", __config__=strict, **fields)
    request = create_model(
        "PredictRequest",
        __config__=strict,
        instances=(list[features], Field(min_length=1, max_length=max_batch)),
    )
    return features, request


class Prediction(BaseModel):
    phishing_probability: float
    label: Literal["phishing", "legitimate"]


class ModelInfo(BaseModel):
    name: str
    version: str
    sha256: str
    signed_by: str


class PredictResponse(BaseModel):
    model: ModelInfo
    predictions: list[Prediction]


async def require_json(request: Request) -> None:
    content_type = request.headers.get("content-type", "")
    if content_type.split(";")[0].strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="content type must be application/json")


def create_app(settings: Settings) -> FastAPI:
    state: dict[str, LoadedModel] = {}
    _, PredictRequest = _request_models(settings.max_batch)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            state["model"] = load_verified_model(settings)
        except Exception as exc:
            log.critical("refusing to start: %s", exc, extra={"bundle": str(settings.bundle)})
            raise
        loaded = state["model"]
        log.info(
            "model verified and loaded",
            extra={
                "model": loaded.model.name,
                "version": loaded.model.version,
                "sha256": loaded.sha256,
                "keyid": loaded.keyid[:16],
            },
        )
        yield
        state.clear()

    docs = settings.enable_docs
    app = FastAPI(
        title="mlseal",
        version="1",
        debug=False,
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )

    @app.middleware("http")
    async def headers_and_access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        response.headers["X-Request-ID"] = request_id
        access.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(1000 * (time.perf_counter() - start), 2),
                **getattr(request.state, "summary", {}),
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default echoes the offending input back; return only where and why.
        errors = [
            {
                "loc": [str(part)[:64] for part in err.get("loc", ())],
                "msg": err.get("msg", ""),
                "type": err.get("type", ""),
            }
            for err in exc.errors()[:20]
        ]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        if "model" not in state:
            return JSONResponse({"status": "not ready"}, status_code=503)
        return JSONResponse({"status": "ready", "model_sha256": state["model"].sha256})

    def model_info() -> ModelInfo:
        loaded = state["model"]
        return ModelInfo(
            name=loaded.model.name,
            version=loaded.model.version,
            sha256=loaded.sha256,
            signed_by=loaded.keyid,
        )

    @app.get("/v1/model")
    async def model_metadata() -> dict[str, Any]:
        statement = state["model"].statement["predicate"]
        return {
            **model_info().model_dump(),
            "features": list(FEATURE_NAMES),
            "source": statement["source"],
            "evaluation": statement["evaluation"],
            "signed_at": statement["signed_at"],
        }

    @app.post("/v1/predict", dependencies=[Depends(require_json)])
    async def predict(request: Request, body: PredictRequest) -> PredictResponse:
        loaded = state["model"]
        X = np.array(
            [[getattr(inst, name) for name in FEATURE_NAMES] for inst in body.instances],
            dtype=np.float64,
        )
        proba = loaded.model.predict_proba(X)
        flagged = proba >= loaded.model.threshold
        request.state.summary = {"instances": len(X), "flagged": int(flagged.sum())}
        return PredictResponse(
            model=model_info(),
            predictions=[
                # Rounded scores: enough for triage, less signal for model extraction.
                Prediction(
                    phishing_probability=round(float(p), 3),
                    label="phishing" if f else "legitimate",
                )
                for p, f in zip(proba, flagged, strict=True)
            ],
        )

    # Outermost first on the wire: size limit, then rate limit, then the app.
    app.add_middleware(RateLimit, per_minute=settings.rate_limit_per_minute)
    app.add_middleware(BodySizeLimit, max_bytes=settings.max_body_bytes)
    return app
