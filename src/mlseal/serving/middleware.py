"""ASGI middleware: security headers and access log, request size limit, per-client rate limit."""

from __future__ import annotations

import ipaddress
import json
import logging
import threading
import time
import uuid
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


class AccessLog:
    """Outermost layer: security headers and one access-log line on every response.

    It wraps the size and rate limits too, so their 413 and 429 rejections carry the same
    headers and reach the log: a throttled client is exactly what the log is there to show.
    A line holds the request id, method, path, status, latency and the counts the endpoint
    leaves in `scope["state"]["summary"]`; never the body, query string or client address.
    """

    def __init__(self, app: Any, headers: dict[str, str], logger: logging.Logger) -> None:
        self.app, self.logger = app, logger
        self.headers = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        self.names = {k for k, _ in self.headers} | {b"x-request-id"}

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        start = time.perf_counter()
        status: int | None = None  # stays None if the client leaves before any response

        async def send_with_headers(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                kept = [h for h in message.get("headers", []) if h[0].lower() not in self.names]
                headers = [*kept, *self.headers, (b"x-request-id", request_id.encode())]
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            status = status or 500  # the server error handler outside answers 500
            raise
        finally:
            self.logger.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status,
                    "duration_ms": round(1000 * (time.perf_counter() - start), 2),
                    **state.get("summary", {}),
                },
            )


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


def client_key(host: str) -> str:
    """Rate-limit bucket for a peer: its IPv4 address, or its IPv6 /64 prefix.

    One IPv6 subscriber usually holds a whole /64, so a bucket per address would hand a
    single client 2^64 fresh quotas. IPv4-mapped IPv6 addresses count as IPv4; anything that
    is not an IP address (a Unix socket peer, a test client) is used as given.
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host
    if ip.version == 6 and ip.ipv4_mapped is None:
        return f"{ipaddress.IPv6Address(int(ip) >> 64 << 64)}/64"  # drops any zone id
    return str(ip.ipv4_mapped if ip.version == 6 else ip)


class RateLimit:
    """Token bucket per client (`client_key`): a cheap brake on scraping and extraction.

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
        client = client_key((scope.get("client") or ("unknown", 0))[0])
        if not self.allow(client):
            await _reject(send, 429, "rate limit exceeded", [(b"retry-after", b"1")])
            return
        await self.app(scope, receive, send)
