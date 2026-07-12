from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from trading_contracts import (
    ArtifactReference,
    ComponentManifest,
    DeploymentManifest,
    OrderIntent,
    PortfolioConstraintState,
    PortfolioIntent,
    ProducerIdentity,
    content_hash,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
NOW = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)
CUTOFF = datetime(2026, 7, 10, 19, 59, 59, tzinfo=timezone.utc)
PRODUCER = ProducerIdentity(name="contract-example-builder", version="0.1.0")


def _base(object_id: str) -> dict:
    return {
        "object_id": object_id,
        "as_of": NOW,
        "producer": PRODUCER,
        "trace_id": "example-trace-001",
        "config_hash": HASH_A,
    }


def _write(path: Path, model) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    output = Path(__file__).resolve().parents[1] / "examples"
    output.mkdir(parents=True, exist_ok=True)

    component = ComponentManifest(
        **_base("component-solusdt-1h-policy"),
        role="asset_policy",
        domain_id="trading_asset_policy_sac_v1",
        artifact=ArtifactReference(
            uri="artifact://sha256/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            content_hash=HASH_B,
            size_bytes=1024,
            media_type="application/x-stable-baselines3",
        ),
        resolved_config={"asset": "crypto:SOL/USDT", "timeframe": "1h"},
        resolved_config_hash=HASH_A,
        code_commits={"agent-multi": "1" * 40},
        dataset_hashes={"training": HASH_A},
        feature_hashes={"technical": HASH_B},
        training_cutoff=CUTOFF,
        seed=7,
        input_contract_versions=["decision_context.v1"],
        output_contract_versions=["asset_intent.v1"],
        validation_metrics={"rap.mean_weekly.v1": 0.001},
        validation_weeks=52,
        execution_contract_version="order_intent.v1",
    )
    _write(output / "component_manifest_asset_policy.json", component)

    portfolio = PortfolioIntent(
        **_base("portfolio-intent-001"),
        target_weights={"crypto:SOL/USDT@1h:technical-v1:sac": 0.35},
        cash_weight=0.65,
        constraints=PortfolioConstraintState(
            gross_exposure_limit=1.0,
            net_exposure_limit=1.0,
            max_cell_weight=0.5,
            turnover_limit=0.5,
        ),
        confidence=0.7,
        allocator_id="static-v1",
    )
    _write(output / "portfolio_intent_smoke.json", portfolio)

    order = OrderIntent(
        **_base("order-intent-001"),
        account_ref="practice-account-ref",
        asset_id="crypto:SOL/USDT",
        venue="research-simulator",
        instrument="SOLUSDT",
        order_type="market",
        delta_units=2.0,
        stop_price=140.0,
        take_profit_price=155.0,
        idempotency_key="example-order-001",
        source_asset_intent_ids=["asset-intent-001"],
        source_portfolio_intent_id=portfolio.object_id,
        preflight={"margin_available": 10000.0, "accepted": True},
    )
    _write(output / "order_intent_smoke.json", order)

    deployment = DeploymentManifest(
        **_base("deployment-balanced-001"),
        release_id="stack-2026w28-balanced-001",
        channel="experimental",
        valid_from=NOW,
        training_cutoff=CUTOFF,
        components={"asset_policy": content_hash(component)},
        portfolio_cells=["crypto:SOL/USDT@1h:technical-v1:sac"],
        validation_metrics={
            "return.mean_weekly.v1": 0.002,
            "rap.mean_weekly.v1": 0.001,
        },
        compatibility={"execution_contract_version": "order_intent.v1"},
    )
    _write(output / "deployment_manifest_smoke.json", deployment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

