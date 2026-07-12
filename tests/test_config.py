from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_contracts import (
    CandidateGenomePatch,
    TradingExperimentConfig,
    TradingRuntimeOverlay,
)


def test_experiment_config_has_stable_hash() -> None:
    first = TradingExperimentConfig(
        experiment={"name": "smoke", "seed": 7},
        data={"asset": "fx:EUR/USD", "timeframe": "4h"},
        asset_policy={"plugin": "project3_sac_actor_critic_agent"},
        training={"total_timesteps": 1000},
    )
    second = TradingExperimentConfig(
        training={"total_timesteps": 1000},
        asset_policy={"plugin": "project3_sac_actor_critic_agent"},
        data={"timeframe": "4h", "asset": "fx:EUR/USD"},
        experiment={"seed": 7, "name": "smoke"},
    )
    assert first.canonical_hash == second.canonical_hash


@pytest.mark.parametrize("key", ["password", "api_key", "private-key", "token"])
def test_experiment_config_rejects_embedded_secrets(key: str) -> None:
    with pytest.raises(ValidationError, match="embedded secrets"):
        TradingExperimentConfig(deployment={key: "do-not-store"})


def test_experiment_config_rejects_unknown_top_level_sections() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TradingExperimentConfig(unknown_section={})


def test_candidate_patch_requires_json_pointers() -> None:
    patch = CandidateGenomePatch(
        base_config_hash="sha256:" + "a" * 64,
        genes={"/risk/rel_volume": 0.08},
    )
    assert patch.genes["/risk/rel_volume"] == 0.08

    with pytest.raises(ValidationError, match="JSON pointers"):
        CandidateGenomePatch(
            base_config_hash="sha256:" + "a" * 64,
            genes={"risk/rel_volume": 0.08},
        )


def test_candidate_patch_requires_canonical_base_hash() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        CandidateGenomePatch(base_config_hash="not-a-hash", genes={})


def test_runtime_overlay_has_independent_stable_hash() -> None:
    overlay = TradingRuntimeOverlay(
        machine_id="omega",
        roots={"DATA_ROOT": "/data", "ARTIFACT_ROOT": "/artifacts"},
        repositories={"agent-multi": "/repos/agent-multi"},
        devices={"default": "cuda:0"},
        environment_refs={"broker": "env:OANDA_TOKEN"},
    )
    reordered = TradingRuntimeOverlay(
        roots={"ARTIFACT_ROOT": "/artifacts", "DATA_ROOT": "/data"},
        machine_id="omega",
        environment_refs={"broker": "env:OANDA_TOKEN"},
        devices={"default": "cuda:0"},
        repositories={"agent-multi": "/repos/agent-multi"},
    )
    assert overlay.canonical_hash == reordered.canonical_hash


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"machine_id": "omega", "roots": {"data": "/data"}}, "root names"),
        (
            {
                "machine_id": "omega",
                "roots": {"DATA_ROOT": "/data"},
                "environment_refs": {"broker": "literal-secret"},
            },
            "env:NAME",
        ),
    ],
)
def test_runtime_overlay_rejects_ambiguous_roots_and_secret_values(
    kwargs: dict,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        TradingRuntimeOverlay(**kwargs)
