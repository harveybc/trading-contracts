"""Execution-era v2 contracts correcting findings AUD-F2-20260801-039..042.

Versioned successors and new families for the L0/L1 demo-trading vertical.
`order_intent.v1` and `execution_report.v1` remain untouched in
``contracts.py``; consumers migrate by version, never by silent edit.

Design constraints encoded here:

- 039: a risk-increasing intent cannot exist without both broker-side
  protective legs; the stop-entry trigger is a separate field from the
  protective stop loss, so the ambiguity cannot be expressed.
- 040: risk is four separate dimensions plus an atomic reservation
  identity, never one overloaded ``rel_volume``.
- 041: the execution lifecycle is a validated state machine with partial,
  cancel, expiry and unknown states, bracket parent/child identity and
  per-leg protection coverage.
- 042: the owner hold/kill command is a deterministic, expiring,
  replay-protected contract that only expresses risk-reducing verbs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .contracts import NonEmpty, PersistedContract, Sha256, StrictModel

FinitePositive = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFraction = Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)]


class ProtectiveBracket(StrictModel):
    """Broker-side protection legs. Never an entry trigger (finding 039)."""

    stop_loss_price: FinitePositive
    take_profit_price: FinitePositive
    trailing_distance: FinitePositive | None = None


class RiskEnvelope(StrictModel):
    """Separate risk dimensions plus atomic reservation (finding 040).

    ``reservation_id`` refers to a worst-case loss-at-stop reservation taken
    atomically before submission and released deterministically on reject,
    partial fill, cancel and close.
    """

    risk_fraction_at_stop: PositiveFraction
    gross_notional_fraction: PositiveFraction
    margin_fraction: PositiveFraction
    daily_loss_budget_fraction: PositiveFraction
    reservation_id: NonEmpty


class OrderIntentV2(PersistedContract):
    schema_version: Literal["order_intent.v2"] = "order_intent.v2"
    account_ref: NonEmpty
    asset_id: NonEmpty
    venue: NonEmpty
    instrument: NonEmpty
    intent_class: Literal["risk_increasing", "risk_reducing"]
    order_type: Literal["market", "limit", "stop"]
    reduce_action: (
        Literal["close", "flatten", "cancel", "modify_protection"] | None
    ) = None
    delta_units: Annotated[float, Field(allow_inf_nan=False)]
    limit_price: FinitePositive | None = None
    entry_trigger_price: FinitePositive | None = None
    protection: ProtectiveBracket | None = None
    risk: RiskEnvelope | None = None
    capability_snapshot_hash: Sha256 | None = None
    idempotency_key: NonEmpty
    source_asset_intent_ids: list[str] = Field(default_factory=list)
    source_portfolio_intent_id: str | None = None
    preflight: dict[str, float | int | bool | str | None] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_protection_contract(self) -> "OrderIntentV2":
        if self.intent_class == "risk_increasing":
            if self.delta_units == 0.0:
                raise ValueError("risk_increasing intent requires nonzero delta_units")
            if self.reduce_action is not None:
                raise ValueError("risk_increasing intent cannot carry reduce_action")
            if self.protection is None:
                raise ValueError(
                    "risk_increasing intent requires broker-side stop loss "
                    "and take profit (finding 039)"
                )
            if self.risk is None:
                raise ValueError("risk_increasing intent requires a RiskEnvelope")
            if self.capability_snapshot_hash is None:
                raise ValueError(
                    "risk_increasing intent requires capability_snapshot_hash"
                )
            if self.order_type == "limit" and self.limit_price is None:
                raise ValueError("limit entry requires limit_price")
            if self.order_type == "stop" and self.entry_trigger_price is None:
                raise ValueError(
                    "stop entry requires entry_trigger_price; the protective "
                    "stop loss is protection.stop_loss_price"
                )
            if self.order_type == "market" and self.entry_trigger_price is not None:
                raise ValueError("market entry cannot carry entry_trigger_price")
            if self.order_type == "limit" and self.entry_trigger_price is not None:
                raise ValueError("limit entry cannot carry entry_trigger_price")
            self._validate_geometry()
        else:
            if self.reduce_action is None:
                raise ValueError("risk_reducing intent requires reduce_action")
            if self.entry_trigger_price is not None:
                raise ValueError("risk_reducing intent cannot carry entry_trigger_price")
            if self.reduce_action == "modify_protection" and self.protection is None:
                raise ValueError("modify_protection requires a ProtectiveBracket")
        return self

    def _validate_geometry(self) -> None:
        assert self.protection is not None
        if (
            self.order_type == "stop"
            and self.entry_trigger_price is not None
            and self.protection.stop_loss_price == self.entry_trigger_price
        ):
            raise ValueError(
                "protective stop loss cannot equal the stop-entry trigger "
                "(finding 039 ambiguity)"
            )
        side_long = self.delta_units > 0
        sl = self.protection.stop_loss_price
        tp = self.protection.take_profit_price
        if side_long and not sl < tp:
            raise ValueError("long entry requires stop_loss_price < take_profit_price")
        if not side_long and not sl > tp:
            raise ValueError("short entry requires stop_loss_price > take_profit_price")
        reference = (
            self.limit_price if self.order_type == "limit" else self.entry_trigger_price
        )
        if reference is not None:
            if side_long and not (sl < reference < tp):
                raise ValueError(
                    "long entry requires stop_loss_price < entry price < "
                    "take_profit_price"
                )
            if not side_long and not (sl > reference > tp):
                raise ValueError(
                    "short entry requires stop_loss_price > entry price > "
                    "take_profit_price"
                )


ExecutionStateV2 = Literal[
    "requested",
    "accepted",
    "partially_filled",
    "filled",
    "rejected",
    "cancel_pending",
    "cancelled",
    "expired",
    "modified",
    "closed",
    "unknown_requires_reconciliation",
]

# Ruling R4 (2026-08-02): includes fill-before-ack, cancel-before-ack,
# expiry-while-cancel-pending, repeated-unknown evidence and reconciliation
# into cancel_pending/modified. `closed` is an ORDER terminal state only;
# open/closed EXPOSURE has its own persisted lifecycle in the LTS ledger and
# is never inferred from an order state (finding 044).
LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    "requested": frozenset(
        {"accepted", "partially_filled", "filled", "cancel_pending",
         "rejected", "cancelled", "expired",
         "unknown_requires_reconciliation"}
    ),
    "accepted": frozenset(
        {"partially_filled", "filled", "cancel_pending", "cancelled", "expired",
         "modified", "unknown_requires_reconciliation"}
    ),
    "partially_filled": frozenset(
        {"partially_filled", "filled", "cancel_pending", "cancelled", "expired",
         "modified", "unknown_requires_reconciliation"}
    ),
    "filled": frozenset({"modified", "closed", "unknown_requires_reconciliation"}),
    "cancel_pending": frozenset(
        {"cancelled", "expired", "partially_filled", "filled",
         "unknown_requires_reconciliation"}
    ),
    "modified": frozenset(
        {"partially_filled", "filled", "cancel_pending", "cancelled", "expired",
         "modified", "closed", "unknown_requires_reconciliation"}
    ),
    "unknown_requires_reconciliation": frozenset(
        {"accepted", "partially_filled", "filled", "rejected", "cancelled",
         "expired", "closed", "cancel_pending", "modified",
         "unknown_requires_reconciliation"}
    ),
    "rejected": frozenset(),
    "cancelled": frozenset(),
    "expired": frozenset(),
    "closed": frozenset(),
}


def is_legal_transition(current: str, target: str) -> bool:
    return target in LEGAL_TRANSITIONS.get(current, frozenset())


class ProtectionLegState(StrictModel):
    leg: Literal["stop_loss", "take_profit"]
    broker_confirmed: bool
    broker_leg_id: str | None = None
    price: FinitePositive | None = None
    covered_units: NonNegativeFinite = 0.0


class ExecutionReportV2(PersistedContract):
    schema_version: Literal["execution_report.v2"] = "execution_report.v2"
    order_intent_id: NonEmpty
    attempt_id: NonEmpty
    bracket_role: Literal["parent", "stop_loss", "take_profit"]
    parent_order_intent_id: str | None = None
    state: ExecutionStateV2
    previous_state: ExecutionStateV2 | None = None
    requested_units: Annotated[float, Field(allow_inf_nan=False)]
    filled_units: NonNegativeFinite = 0.0
    requested_price: FinitePositive | None = None
    filled_price: FinitePositive | None = None
    protection_legs: list[ProtectionLegState] = Field(default_factory=list)
    spread_cost: NonNegativeFinite = 0.0
    slippage_cost: NonNegativeFinite = 0.0
    commission: NonNegativeFinite = 0.0
    financing: Annotated[float, Field(allow_inf_nan=False)] = 0.0
    conversion_cost: NonNegativeFinite = 0.0
    broker_ids: dict[str, str] = Field(default_factory=dict)
    latency_ms: NonNegativeFinite | None = None
    reconciliation_required: bool = False

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "ExecutionReportV2":
        magnitude = abs(self.requested_units)
        if self.filled_units > magnitude:
            raise ValueError("filled_units cannot exceed |requested_units|")
        if self.state == "partially_filled" and not (
            0.0 < self.filled_units < magnitude
        ):
            raise ValueError(
                "partially_filled requires 0 < filled_units < |requested_units|"
            )
        if self.state == "filled" and self.filled_units != magnitude:
            raise ValueError("filled requires filled_units == |requested_units|")
        if self.bracket_role != "parent" and not self.parent_order_intent_id:
            raise ValueError("child protection leg requires parent_order_intent_id")
        if self.previous_state is not None and not is_legal_transition(
            self.previous_state, self.state
        ):
            raise ValueError(
                f"illegal lifecycle transition {self.previous_state!r} -> "
                f"{self.state!r}"
            )
        if (self.state == "unknown_requires_reconciliation") != (
            self.reconciliation_required
        ):
            raise ValueError(
                "reconciliation_required must be true exactly when state is "
                "unknown_requires_reconciliation"
            )
        return self


def protection_covers_filled(report: ExecutionReportV2) -> bool:
    """True when broker-confirmed SL and TP legs each cover filled_units.

    The L0/L1 services must treat a filled parent for which this is false as
    unprotected exposure requiring immediate flatten-and-alert (doc 22 §5).
    """
    if report.bracket_role != "parent" or report.filled_units == 0.0:
        return True
    coverage = {"stop_loss": 0.0, "take_profit": 0.0}
    for leg in report.protection_legs:
        if leg.broker_confirmed:
            coverage[leg.leg] += leg.covered_units
    return all(
        covered >= report.filled_units - 1e-12 for covered in coverage.values()
    )


class InstrumentCapability(StrictModel):
    instrument: NonEmpty
    tradeable: bool
    shortable: bool
    min_units: FinitePositive
    unit_step: FinitePositive
    price_decimals: Annotated[int, Field(ge=0, le=12)]
    margin_rate: PositiveFraction | None = None
    native_stop_loss: bool
    native_take_profit: bool
    native_bracket: bool


class BrokerCapabilitySnapshot(PersistedContract):
    """Account-specific capability facts; referenced by hash from intents.

    ``account_fingerprint`` is a one-way fingerprint; raw account identifiers
    never enter persisted contracts (doc 09 §5).

    Ruling R3 (2026-08-02, amended pre-consumption with auditor order):
    ``capability_evidence`` is mandatory provenance. ``synthetic_fixture``
    may drive L0 mechanics only, requires a synthetic fingerprint, and is
    mechanically excluded from venue-readiness, broker-compatibility and L1
    authorization claims. ``recorded_observed`` is replay evidence, never
    current readiness. Only fresh ``live_observed`` evidence from the target
    account supports a readiness claim. Capabilities are never inferred
    across venues.
    """

    schema_version: Literal["broker_capability_snapshot.v1"] = (
        "broker_capability_snapshot.v1"
    )
    venue: NonEmpty
    account_fingerprint: NonEmpty
    environment: Literal["paper", "demo", "live"]
    capability_evidence: Literal[
        "live_observed", "recorded_observed", "synthetic_fixture"
    ]
    source_artifact_hash: Sha256
    source_observed_at: datetime
    instruments: list[InstrumentCapability]

    @model_validator(mode="after")
    def validate_capability_provenance(self) -> "BrokerCapabilitySnapshot":
        names = [entry.instrument for entry in self.instruments]
        if len(names) != len(set(names)):
            raise ValueError("duplicate instrument in capability snapshot")
        if (
            self.source_observed_at.tzinfo is None
            or self.source_observed_at.utcoffset() is None
        ):
            raise ValueError("source_observed_at must be timezone-aware")
        if self.capability_evidence == "synthetic_fixture" and not (
            self.account_fingerprint.startswith("synthetic-")
        ):
            raise ValueError(
                "synthetic_fixture capability requires a synthetic account "
                "fingerprint (prefix 'synthetic-')"
            )
        return self


class OwnerCommand(PersistedContract):
    """Deterministic owner hold/kill contract (finding 042).

    Only risk-reducing verbs are expressible. The handler additionally
    enforces an issuer allowlist, exact-phrase match, nonce persistence and
    expiry; no LLM, Hermes or social process may originate, transform or
    approve an instance.
    """

    schema_version: Literal["owner_command.v1"] = "owner_command.v1"
    command: Literal["hold", "kill", "flatten_all", "cancel_pending"]
    issuer_id: NonEmpty
    exact_phrase: NonEmpty
    nonce: NonEmpty
    expires_at: datetime
    idempotency_key: NonEmpty

    @model_validator(mode="after")
    def validate_expiry(self) -> "OwnerCommand":
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.expires_at <= self.as_of:
            raise ValueError("expires_at must be after as_of")
        return self
