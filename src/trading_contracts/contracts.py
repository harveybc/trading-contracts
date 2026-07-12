from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Sha256 = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
NonEmpty = Annotated[str, Field(min_length=1)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProducerIdentity(StrictModel):
    name: NonEmpty
    version: NonEmpty
    instance_id: str | None = None


class PersistedContract(StrictModel):
    schema_version: NonEmpty
    object_id: NonEmpty
    as_of: datetime
    valid_until: datetime | None = None
    producer: ProducerIdentity
    trace_id: NonEmpty
    config_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_time_contract(self) -> "PersistedContract":
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        if self.valid_until is not None:
            if self.valid_until.tzinfo is None or self.valid_until.utcoffset() is None:
                raise ValueError("valid_until must be timezone-aware")
            if self.valid_until < self.as_of:
                raise ValueError("valid_until cannot precede as_of")
        return self


class AssetClass(str, Enum):
    FX = "fx"
    CRYPTO = "crypto"
    EQUITY = "equity"
    ETF = "etf"
    COMMODITY = "commodity"
    INDEX = "index"
    FIXED_INCOME = "fixed_income"
    CASH = "cash"
    OTHER = "other"


class AssetDefinition(PersistedContract):
    schema_version: Literal["asset_definition.v1"] = "asset_definition.v1"
    asset_id: NonEmpty
    asset_class: AssetClass
    base: NonEmpty
    quote: NonEmpty
    venue_mappings: dict[str, str] = Field(default_factory=dict)


def make_cell_id(
    asset_id: str,
    timeframe: str,
    data_profile: str,
    policy_role: str,
) -> str:
    return f"{asset_id}@{timeframe}:{data_profile}:{policy_role}"


class CellDefinition(PersistedContract):
    schema_version: Literal["cell_definition.v1"] = "cell_definition.v1"
    cell_id: NonEmpty
    asset_id: NonEmpty
    timeframe: NonEmpty
    data_profile: NonEmpty
    policy_role: NonEmpty

    @model_validator(mode="after")
    def validate_cell_id(self) -> "CellDefinition":
        expected = make_cell_id(
            self.asset_id,
            self.timeframe,
            self.data_profile,
            self.policy_role,
        )
        if self.cell_id != expected:
            raise ValueError(f"cell_id must be {expected!r}")
        return self


class ContextToken(StrictModel):
    token_type: NonEmpty
    event_time: datetime
    values: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    observed: bool = True

    @model_validator(mode="after")
    def validate_event_time(self) -> "ContextToken":
        if self.event_time.tzinfo is None or self.event_time.utcoffset() is None:
            raise ValueError("context token event_time must be timezone-aware")
        return self


class MarketSnapshot(PersistedContract):
    schema_version: Literal["market_snapshot.v1"] = "market_snapshot.v1"
    asset_id: NonEmpty
    timeframe: NonEmpty
    local_features: dict[str, float | int | bool | None] = Field(default_factory=dict)
    feature_freshness_seconds: dict[str, float] = Field(default_factory=dict)
    context_tokens: list[ContextToken] = Field(default_factory=list)
    context_mask: list[bool] = Field(default_factory=list)
    cross_asset_state: dict[str, dict[str, float | int | bool | None]] = Field(
        default_factory=dict
    )
    calendar_state: dict[str, str | float | int | bool | None] = Field(
        default_factory=dict
    )
    venue_state: dict[str, str | float | int | bool | None] = Field(
        default_factory=dict
    )
    data_manifest_hash: Sha256
    feature_manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_context_mask(self) -> "MarketSnapshot":
        if self.context_mask and len(self.context_mask) != len(self.context_tokens):
            raise ValueError("context_mask length must match context_tokens")
        return self


class PredictionHorizon(StrictModel):
    outputs: dict[str, float | int | bool | str | None]
    confidence: Probability | None = None
    output_schema: NonEmpty


class PredictionBundle(PersistedContract):
    schema_version: Literal["prediction_bundle.v1"] = "prediction_bundle.v1"
    asset_id: NonEmpty
    horizons: dict[str, PredictionHorizon]
    model_artifact_hash: Sha256


class PositionState(StrictModel):
    side: Literal["flat", "long", "short"] = "flat"
    units: float = 0.0
    average_price: float | None = None
    unrealized_return: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0


class DecisionContext(PersistedContract):
    schema_version: Literal["decision_context.v1"] = "decision_context.v1"
    cell_id: NonEmpty
    market: MarketSnapshot
    predictions: list[PredictionBundle] = Field(default_factory=list)
    position: PositionState = Field(default_factory=PositionState)
    pending_intent_ids: list[str] = Field(default_factory=list)
    cell_risk_budget: Annotated[float, Field(ge=0.0)] = 0.0
    portfolio_risk_budget: Annotated[float, Field(ge=0.0)] = 0.0
    rush_probability: Probability | None = None
    regime: str | None = None
    seconds_to_weekend_close: float | None = None
    previous_action: str | None = None
    previous_execution_id: str | None = None


class RiskGeometry(StrictModel):
    mode: NonEmpty
    k_sl: Annotated[float, Field(gt=0.0)] | None = None
    k_tp: Annotated[float, Field(gt=0.0)] | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    trailing_distance: float | None = None


class AssetAction(str, Enum):
    TARGET = "target"
    HOLD = "hold"
    CLOSE = "close"
    MODIFY_PROTECTION = "modify_protection"
    NO_TRADE = "no_trade"


class AssetIntent(PersistedContract):
    schema_version: Literal["asset_intent.v1"] = "asset_intent.v1"
    cell_id: NonEmpty
    asset_id: NonEmpty
    action: AssetAction
    target_exposure: Annotated[float, Field(ge=-1.0, le=1.0)] | None = None
    confidence: Probability | None = None
    strategy_rel_volume: Annotated[float, Field(ge=0.0)] | None = None
    risk_geometry: RiskGeometry | None = None
    reason_codes: list[str] = Field(default_factory=list)
    artifact_hash: Sha256

    @model_validator(mode="after")
    def validate_target(self) -> "AssetIntent":
        if self.action == AssetAction.TARGET and self.target_exposure is None:
            raise ValueError("target action requires target_exposure")
        return self


class PortfolioConstraintState(StrictModel):
    gross_exposure_limit: Annotated[float, Field(ge=0.0)]
    net_exposure_limit: Annotated[float, Field(ge=0.0)]
    max_cell_weight: Annotated[float, Field(ge=0.0, le=1.0)]
    turnover_limit: Annotated[float, Field(ge=0.0)] | None = None


class PortfolioIntent(PersistedContract):
    schema_version: Literal["portfolio_intent.v1"] = "portfolio_intent.v1"
    target_weights: dict[str, float]
    cash_weight: float
    constraints: PortfolioConstraintState
    confidence: Probability | None = None
    allocator_id: NonEmpty
    allocator_artifact_hash: Sha256 | None = None


class OrderIntent(PersistedContract):
    schema_version: Literal["order_intent.v1"] = "order_intent.v1"
    account_ref: NonEmpty
    asset_id: NonEmpty
    venue: NonEmpty
    instrument: NonEmpty
    order_type: Literal["market", "limit", "stop"]
    delta_units: float
    limit_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    trailing_distance: float | None = None
    idempotency_key: NonEmpty
    source_asset_intent_ids: list[str] = Field(default_factory=list)
    source_portfolio_intent_id: str | None = None
    preflight: dict[str, float | int | bool | str | None] = Field(default_factory=dict)


class AccountSnapshot(StrictModel):
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    currency: NonEmpty


class ExecutionReport(PersistedContract):
    schema_version: Literal["execution_report.v1"] = "execution_report.v1"
    order_intent_id: NonEmpty
    state: Literal["requested", "accepted", "filled", "rejected", "modified", "closed"]
    requested_units: float
    filled_units: float = 0.0
    requested_price: float | None = None
    filled_price: float | None = None
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission: float = 0.0
    financing: float = 0.0
    conversion_cost: float = 0.0
    broker_ids: dict[str, str] = Field(default_factory=dict)
    latency_ms: float | None = None
    account: AccountSnapshot | None = None
    reason_code: str | None = None


class ArtifactReference(StrictModel):
    uri: NonEmpty
    content_hash: Sha256
    size_bytes: Annotated[int, Field(ge=0)]
    media_type: NonEmpty


class ComponentManifest(PersistedContract):
    schema_version: Literal["component_manifest.v1"] = "component_manifest.v1"
    role: NonEmpty
    domain_id: NonEmpty
    artifact: ArtifactReference
    resolved_config: dict[str, Any]
    resolved_config_hash: Sha256
    code_commits: dict[str, NonEmpty]
    dataset_hashes: dict[str, Sha256]
    feature_hashes: dict[str, Sha256] = Field(default_factory=dict)
    training_cutoff: datetime
    seed: int
    input_contract_versions: list[str]
    output_contract_versions: list[str]
    validation_metrics: dict[str, float]
    validation_weeks: Annotated[int, Field(ge=0)]
    execution_contract_version: NonEmpty

    @model_validator(mode="after")
    def validate_training_cutoff(self) -> "ComponentManifest":
        if self.training_cutoff.tzinfo is None or self.training_cutoff.utcoffset() is None:
            raise ValueError("training_cutoff must be timezone-aware")
        return self


class DeploymentManifest(PersistedContract):
    schema_version: Literal["deployment_manifest.v1"] = "deployment_manifest.v1"
    release_id: NonEmpty
    channel: Literal["stable", "adaptive", "experimental", "pinned"]
    valid_from: datetime
    training_cutoff: datetime
    components: dict[str, Sha256]
    portfolio_cells: list[str]
    validation_metrics: dict[str, float]
    compatibility: dict[str, str | float | int | bool]
    rollback_release_id: str | None = None
    signature: str | None = None

    @model_validator(mode="after")
    def validate_deployment_cutoff(self) -> "DeploymentManifest":
        for name, value in (
            ("valid_from", self.valid_from),
            ("training_cutoff", self.training_cutoff),
        ):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.training_cutoff > self.valid_from:
            raise ValueError("training_cutoff cannot exceed valid_from")
        return self


class MetricDefinition(StrictModel):
    metric_key: Annotated[str, Field(pattern=r"^[a-z0-9_.]+\.v[0-9]+$")]
    label: NonEmpty
    description: NonEmpty
    unit: Literal["fraction", "percent", "currency", "seconds", "count", "ratio"]
    period: NonEmpty
    aggregation: Literal["mean", "compound", "sum", "max", "min", "last", "quantile"]
    direction: Literal["higher", "lower", "constraint_only"]
    provenance_classes: list[
        Literal["train", "validation", "test", "synthetic", "shadow", "live"]
    ]
    formula: NonEmpty
    denominator: NonEmpty
    coverage_rule: NonEmpty
    required: bool = False
    nullable: bool = True
    failure_semantics: NonEmpty


class MetricCatalog(StrictModel):
    schema_version: Literal["metric_catalog.v1"] = "metric_catalog.v1"
    catalog_id: NonEmpty
    metrics: list[MetricDefinition]

    @model_validator(mode="after")
    def validate_unique_metric_keys(self) -> "MetricCatalog":
        keys = [metric.metric_key for metric in self.metrics]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError("duplicate metric keys: " + ", ".join(duplicates))
        return self
