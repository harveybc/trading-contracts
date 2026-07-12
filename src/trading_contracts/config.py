from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, model_validator

from .canonical import content_hash
from .contracts import Sha256, StrictModel


_SECRET_KEYS = {
    "api_key",
    "api_secret",
    "password",
    "private_key",
    "secret",
    "token",
}


def _embedded_secret_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}/{key}"
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SECRET_KEYS and item not in (None, ""):
                found.append(path)
            found.extend(_embedded_secret_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_embedded_secret_paths(item, f"{prefix}/{index}"))
    return found


class TradingExperimentConfig(StrictModel):
    schema_version: Literal["trading_experiment.v1"] = "trading_experiment.v1"
    experiment: dict[str, Any] = Field(default_factory=dict)
    code: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    walk_forward: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    context_encoder: dict[str, Any] = Field(default_factory=dict)
    predictions: dict[str, Any] = Field(default_factory=dict)
    rush_detector: dict[str, Any] = Field(default_factory=dict)
    asset_policy: dict[str, Any] = Field(default_factory=dict)
    lifecycle_policy: dict[str, Any] = Field(default_factory=dict)
    risk: dict[str, Any] = Field(default_factory=dict)
    portfolio: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    objectives: dict[str, Any] = Field(default_factory=dict)
    optimization: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    olap: dict[str, Any] = Field(default_factory=dict)
    deployment: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_embedded_secrets(self) -> "TradingExperimentConfig":
        paths = _embedded_secret_paths(self.model_dump(mode="python"))
        if paths:
            raise ValueError("embedded secrets are forbidden at: " + ", ".join(paths))
        return self

    @property
    def canonical_hash(self) -> str:
        return content_hash(self)


class CandidateGenomePatch(StrictModel):
    schema_version: Literal["candidate_genome_patch.v1"] = "candidate_genome_patch.v1"
    base_config_hash: Sha256
    genes: dict[str, Any]

    @model_validator(mode="after")
    def validate_json_pointers(self) -> "CandidateGenomePatch":
        invalid = [pointer for pointer in self.genes if not pointer.startswith("/")]
        if invalid:
            raise ValueError("gene keys must be JSON pointers: " + ", ".join(invalid))
        return self


_ROOT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*_ROOT$")
_ENV_REFERENCE = re.compile(r"^env:[A-Z_][A-Z0-9_]*$")


class TradingRuntimeOverlay(StrictModel):
    schema_version: Literal["trading_runtime_overlay.v1"] = "trading_runtime_overlay.v1"
    machine_id: str
    roots: dict[str, str]
    repositories: dict[str, str] = Field(default_factory=dict)
    devices: dict[str, str] = Field(default_factory=dict)
    resource_limits: dict[str, int | float] = Field(default_factory=dict)
    environment_refs: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_runtime_overlay(self) -> "TradingRuntimeOverlay":
        if not self.machine_id.strip():
            raise ValueError("machine_id cannot be empty")
        invalid_roots = sorted(name for name in self.roots if not _ROOT_NAME.fullmatch(name))
        if invalid_roots:
            raise ValueError("invalid runtime root names: " + ", ".join(invalid_roots))
        empty_paths = sorted(
            name
            for name, path in {**self.roots, **self.repositories}.items()
            if not str(path).strip()
        )
        if empty_paths:
            raise ValueError("runtime paths cannot be empty: " + ", ".join(empty_paths))
        invalid_refs = sorted(
            name
            for name, reference in self.environment_refs.items()
            if not _ENV_REFERENCE.fullmatch(reference)
        )
        if invalid_refs:
            raise ValueError(
                "environment_refs must contain env:NAME references: "
                + ", ".join(invalid_refs)
            )
        return self

    @property
    def canonical_hash(self) -> str:
        return content_hash(self)
