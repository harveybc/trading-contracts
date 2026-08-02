from __future__ import annotations

import json
from pathlib import Path

from trading_contracts import (
    AssetDefinition,
    AssetIntent,
    BrokerCapabilitySnapshot,
    CandidateGenomePatch,
    CellDefinition,
    ComponentManifest,
    DecisionContext,
    DeploymentManifest,
    ExecutionReport,
    ExecutionReportV2,
    MarketSnapshot,
    MetricCatalog,
    OrderIntent,
    OrderIntentV2,
    OwnerCommand,
    PortfolioIntent,
    PredictionBundle,
    TradingExperimentConfig,
    TradingRuntimeOverlay,
)


MODELS = (
    AssetDefinition,
    CellDefinition,
    MarketSnapshot,
    MetricCatalog,
    PredictionBundle,
    DecisionContext,
    AssetIntent,
    PortfolioIntent,
    OrderIntent,
    OrderIntentV2,
    ExecutionReport,
    ExecutionReportV2,
    BrokerCapabilitySnapshot,
    OwnerCommand,
    ComponentManifest,
    DeploymentManifest,
    TradingExperimentConfig,
    TradingRuntimeOverlay,
    CandidateGenomePatch,
)


def main() -> int:
    output_dir = Path(__file__).resolve().parents[1] / "schemas"
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        path = output_dir / f"{model.__name__}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
