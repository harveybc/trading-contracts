from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_contracts import (
    AssetAction,
    AssetClass,
    AssetDefinition,
    AssetIntent,
    CellDefinition,
    ContextToken,
    MarketSnapshot,
    MetricCatalog,
    PredictionBundle,
    PredictionHorizon,
    ProducerIdentity,
    RiskGeometry,
    canonical_json,
    content_hash,
    make_cell_id,
)


NOW = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
PRODUCER = ProducerIdentity(name="contract-test", version="1.0.0")


def base_fields(object_id: str) -> dict:
    return {
        "object_id": object_id,
        "as_of": NOW,
        "producer": PRODUCER,
        "trace_id": "trace-001",
        "config_hash": HASH_A,
    }


def test_asset_and_cell_identity_are_explicit() -> None:
    asset = AssetDefinition(
        **base_fields("asset-eurusd"),
        asset_id="fx:EUR/USD",
        asset_class=AssetClass.FX,
        base="EUR",
        quote="USD",
        venue_mappings={"oanda": "EUR_USD"},
    )
    cell_id = make_cell_id(asset.asset_id, "4h", "technical-v1", "sac")
    cell = CellDefinition(
        **base_fields("cell-eurusd-4h"),
        cell_id=cell_id,
        asset_id=asset.asset_id,
        timeframe="4h",
        data_profile="technical-v1",
        policy_role="sac",
    )
    assert cell.cell_id == "fx:EUR/USD@4h:technical-v1:sac"

    with pytest.raises(ValidationError, match="cell_id must be"):
        CellDefinition(
            **base_fields("bad-cell"),
            cell_id="wrong",
            asset_id=asset.asset_id,
            timeframe="4h",
            data_profile="technical-v1",
            policy_role="sac",
        )


def test_market_prediction_and_asset_intent_round_trip() -> None:
    market = MarketSnapshot(
        **base_fields("snapshot-001"),
        asset_id="crypto:SOL/USDT",
        timeframe="1h",
        local_features={"return_1": 0.01, "atr": 2.3},
        context_tokens=[
            ContextToken(token_type="bar", event_time=NOW, values={"close": 145.0})
        ],
        context_mask=[True],
        data_manifest_hash=HASH_A,
        feature_manifest_hash=HASH_B,
    )
    prediction = PredictionBundle(
        **base_fields("prediction-001"),
        asset_id=market.asset_id,
        horizons={
            "6h": PredictionHorizon(
                outputs={"up_probability": 0.7},
                confidence=0.8,
                output_schema="direction_probability.v1",
            )
        },
        model_artifact_hash=HASH_B,
    )
    intent = AssetIntent(
        **base_fields("intent-001"),
        cell_id="crypto:SOL/USDT@1h:context-v1:sac",
        asset_id=market.asset_id,
        action=AssetAction.TARGET,
        target_exposure=0.4,
        confidence=prediction.horizons["6h"].confidence,
        urgency=0.75,
        strategy_rel_volume=0.05,
        risk_geometry=RiskGeometry(mode="atr", k_sl=2.0, k_tp=3.0),
        reason_codes=["long_horizon_edge"],
        artifact_hash=HASH_A,
    )
    restored = AssetIntent.model_validate_json(intent.model_dump_json())
    assert restored == intent


@pytest.mark.parametrize("urgency", [0.0, 1.0])
def test_asset_intent_accepts_urgency_bounds(urgency: float) -> None:
    intent = AssetIntent(
        **base_fields(f"intent-urgency-{urgency}"),
        cell_id="fx:EUR/USD@4h:technical-v1:sac",
        asset_id="fx:EUR/USD",
        action=AssetAction.HOLD,
        urgency=urgency,
        artifact_hash=HASH_A,
    )
    assert intent.urgency == urgency


def test_asset_intent_urgency_defaults_to_none() -> None:
    intent = AssetIntent(
        **base_fields("intent-without-urgency"),
        cell_id="fx:EUR/USD@4h:technical-v1:sac",
        asset_id="fx:EUR/USD",
        action=AssetAction.HOLD,
        artifact_hash=HASH_A,
    )
    assert intent.urgency is None


@pytest.mark.parametrize("urgency", [-0.01, 1.01])
def test_asset_intent_rejects_urgency_outside_bounds(urgency: float) -> None:
    with pytest.raises(ValidationError):
        AssetIntent(
            **base_fields(f"intent-invalid-urgency-{urgency}"),
            cell_id="fx:EUR/USD@4h:technical-v1:sac",
            asset_id="fx:EUR/USD",
            action=AssetAction.HOLD,
            urgency=urgency,
            artifact_hash=HASH_A,
        )


def test_contracts_reject_unknown_fields_and_naive_dates() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AssetDefinition(
            **base_fields("asset"),
            asset_id="fx:EUR/USD",
            asset_class="fx",
            base="EUR",
            quote="USD",
            unexpected=True,
        )

    fields = base_fields("asset")
    fields["as_of"] = datetime(2026, 7, 10, 20, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        AssetDefinition(
            **fields,
            asset_id="fx:EUR/USD",
            asset_class="fx",
            base="EUR",
            quote="USD",
        )


def test_context_mask_must_match_tokens() -> None:
    with pytest.raises(ValidationError, match="context_mask length"):
        MarketSnapshot(
            **base_fields("snapshot"),
            asset_id="fx:EUR/USD",
            timeframe="4h",
            context_tokens=[],
            context_mask=[True],
            data_manifest_hash=HASH_A,
            feature_manifest_hash=HASH_B,
        )


def test_target_action_requires_exposure() -> None:
    with pytest.raises(ValidationError, match="requires target_exposure"):
        AssetIntent(
            **base_fields("intent"),
            cell_id="fx:EUR/USD@4h:technical-v1:sac",
            asset_id="fx:EUR/USD",
            action="target",
            artifact_hash=HASH_A,
        )


def test_canonical_hash_is_order_independent_and_finite() -> None:
    left = {"b": [2, 1], "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "b": [2, 1]}
    assert canonical_json(left) == canonical_json(right)
    assert content_hash(left) == content_hash(right)
    assert content_hash(left).startswith("sha256:")

    with pytest.raises(ValueError, match="NaN or infinity"):
        canonical_json({"bad": float("nan")})


def test_metric_catalog_fixture_is_valid_and_unique() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "metric_catalog_v1.json"
    catalog = MetricCatalog.model_validate(json.loads(path.read_text(encoding="utf-8")))
    keys = {metric.metric_key for metric in catalog.metrics}
    assert "return.mean_weekly.v1" in keys
    assert "return.compounded_annual.v1" in keys
    assert "rap.mean_weekly.v1" in keys
    assert "rap.annual_sum.v1" in keys
    assert len(keys) == len(catalog.metrics)
