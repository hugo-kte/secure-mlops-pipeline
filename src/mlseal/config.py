"""Pipeline configuration: one YAML file, validated strictly (unknown keys are errors)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Share = Annotated[float, Field(ge=0.0, le=1.0)]


class DataPolicy(_Strict):
    """Limits the data gate enforces before any training happens."""

    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    min_rows: int = Field(gt=0)
    max_rows: int = Field(gt=0)
    phishing_share: tuple[Share, Share]
    max_duplicate_share: Share
    max_conflict_share: Share
    max_feature_psi: float = Field(gt=0.0)


class UpstreamConfig(_Strict):
    """Where the pinned file comes from, to re-check it against the publisher."""

    url: Annotated[str, Field(pattern=r"^https://")]
    member: str
    archive_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class DataConfig(_Strict):
    path: Path
    reference_profile: Path
    policy: DataPolicy
    upstream: UpstreamConfig | None = None


class SplitConfig(_Strict):
    test_size: float = Field(gt=0.0, lt=0.5)
    cv_folds: int = Field(ge=3, le=20)


class LogisticSpec(_Strict):
    kind: Literal["logistic_regression"]
    C: float = Field(gt=0.0)


class BoostingSpec(_Strict):
    kind: Literal["gradient_boosting"]
    n_estimators: int = Field(ge=1, le=1000)
    max_depth: int = Field(ge=1, le=8)
    learning_rate: float = Field(gt=0.0, le=1.0)
    subsample: float = Field(default=1.0, gt=0.0, le=1.0)


CandidateSpec = Annotated[LogisticSpec | BoostingSpec, Field(discriminator="kind")]


class ModelConfig(_Strict):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    candidates: list[CandidateSpec] = Field(min_length=1)
    selection_metric: Literal["roc_auc", "f1", "accuracy"]


class EvaluationConfig(_Strict):
    threshold: float = Field(gt=0.0, lt=1.0)
    bootstrap: int = Field(ge=100, le=100_000)


Metric = Literal["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]


class GatePolicy(_Strict):
    """Release criteria on the held-out (de-duplicated) test set."""

    minimum: dict[Metric, Share]
    max_regression: dict[Metric, Share]


class PipelineConfig(_Strict):
    seed: int
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    gates: GatePolicy


def load_config(path: Path) -> PipelineConfig:
    import yaml  # training-side dependency; never needed by the server

    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)  # safe_load: plain data only, no Python object tags
    return PipelineConfig.model_validate(raw)
