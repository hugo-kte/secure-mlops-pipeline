"""ASGI middleware: request size limit and per-client rate limiting."""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any

Scope = dict[str, Any]
Message = dict[str, Any]


async def _reject(send: Any, status: int, detail: str, headers: list | None = None) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                *(headers or []),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BodySizeLimit:
    """Read at most `max_bytes` of body before the app sees anything; 413 beyond that.

    The body is buffered here (it is small by construction) and replayed to the app, so the
    limit holds whether Content-Length is honest, absent (chunked) or a lie.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            await _reject(send, 413, "request body too large")
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":  # client disconnected
                return
            body += message.get("body", b"")
            if len(body) > self.max_bytes:
                await _reject(send, 413, "request body too large")
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()  # later calls only wait for the disconnect
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


class RateLimit:
    """Token bucket per client address: a cheap brake on scraping and model extraction.

    In-process only; a deployment with several replicas enforces quotas at the gateway too.
    The number of tracked clients is bounded so the limiter itself cannot exhaust memory.
    """

    def __init__(self, app: Any, per_minute: int, max_clients: int = 10_000) -> None:
        self.app = app
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.max_clients = max_clients
        self.buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self.lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self.lock:
            tokens, last = self.buckets.pop(client, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            allowed = tokens >= 1.0
            self.buckets[client] = (tokens - 1.0 if allowed else tokens, now)
            while len(self.buckets) > self.max_clients:
                self.buckets.popitem(last=False)
            return allowed

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or self.capacity <= 0 or not scope["path"].startswith("/v1/"):
            await self.app(scope, receive, send)
            return
        client = (scope.get("client") or ("unknown", 0))[0]
        if not self.allow(client):
            await _reject(send, 429, "rate limit exceeded", [(b"retry-after", b"1")])
            return
        await self.app(scope, receive, send)
