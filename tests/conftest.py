"""Shared fixtures: one small, fast training run on the real dataset per test session."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

os.environ.setdefault("MLSEAL_N_JOBS", "2")

from mlseal.config import load_config
from mlseal.gates import run_gates, summary
from mlseal.pipeline import train
from mlseal.provenance import build_statement
from mlseal.signing import generate_key, save_keypair, sign_bundle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "phishing_websites.arff"
CONFIG = ROOT / "configs" / "pipeline.yaml"

FAST_CANDIDATES = [
    {"kind": "logistic_regression", "C": 1.0},
    {"kind": "gradient_boosting", "n_estimators": 40, "max_depth": 3, "learning_rate": 0.2},
]


def write_config(directory: Path, data_path: Path = DATA, **sections: Any) -> Path:
    """The real configuration with absolute paths, fast candidates and optional overrides."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["data"]["path"] = str(data_path)
    raw["data"]["reference_profile"] = str(ROOT / "configs" / "reference_profile.json")
    raw["model"]["candidates"] = FAST_CANDIDATES
    raw["evaluation"]["bootstrap"] = 200
    for section, values in sections.items():
        raw[section] = {**raw[section], **values}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pipeline.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@dataclass(frozen=True)
class Trained:
    bundle: Path
    config_path: Path
    metrics: dict[str, Any]


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    base = tmp_path_factory.mktemp("trained")
    config_path = write_config(base)
    cfg = load_config(config_path)
    bundle = base / "bundle"
    metrics = train(
        cfg, bundle, config_path=config_path, lock_path=ROOT / "uv.lock", log=lambda _: None
    )
    report = summary(run_gates(metrics, cfg.gates, None), None)
    (bundle / "gates.json").write_text(json.dumps(report), encoding="utf-8")
    return Trained(bundle=bundle, config_path=config_path, metrics=metrics)


@dataclass(frozen=True)
class Keys:
    private: Path
    public: Path


@pytest.fixture(scope="session")
def keys(tmp_path_factory: pytest.TempPathFactory) -> Keys:
    private, public = save_keypair(generate_key(), tmp_path_factory.mktemp("keys") / "k")
    return Keys(private=private, public=public)


@pytest.fixture(scope="session")
def signed_bundle(tmp_path_factory: pytest.TempPathFactory, trained: Trained, keys: Keys) -> Path:
    from mlseal.signing import load_private_key

    bundle = tmp_path_factory.mktemp("signed") / "bundle"
    shutil.copytree(trained.bundle, bundle)
    sign_bundle(bundle, build_statement(bundle), load_private_key(keys.private))
    return bundle


@pytest.fixture
def bundle_copy(tmp_path: Path, signed_bundle: Path) -> Path:
    """A private, writable copy of the signed bundle for tampering tests."""
    target = tmp_path / "bundle"
    shutil.copytree(signed_bundle, target)
    return target
