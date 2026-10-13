"""Server settings from the environment (12-factor), validated at startup."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(name)
    value = default if raw is None else int(raw)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be within {lo}..{hi}")
    return value


@dataclass(frozen=True)
class Settings:
    bundle: Path
    public_keys: tuple[Path, ...]
    max_body_bytes: int = 32 * 1024  # 32 instances are ~23 KiB
    max_batch: int = 32
    rate_limit_per_minute: int = 600  # per IPv4 address or IPv6 /64; 0 disables
    enable_docs: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        keys = tuple(Path(p) for p in env.get("MLSEAL_PUBLIC_KEY", "").split(os.pathsep) if p)
        return cls(
            bundle=Path(env.get("MLSEAL_BUNDLE", "build/bundle")),
            public_keys=keys,
            max_body_bytes=_int(env, "MLSEAL_MAX_BODY_BYTES", 32 * 1024, 256, 1024 * 1024),
            max_batch=_int(env, "MLSEAL_MAX_BATCH", 32, 1, 1024),
            rate_limit_per_minute=_int(env, "MLSEAL_RATE_LIMIT_PER_MINUTE", 600, 0, 1_000_000),
            enable_docs=env.get("MLSEAL_ENABLE_DOCS", "0") == "1",
        )
