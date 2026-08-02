"""Adversarial regressions for findings AUD-F2-20260801-039..042 (contract layer).

Each test names the failure it proves impossible at the schema boundary.
Service-level fixtures (reservations, replay, restart, sink) live in `lts`.
"""
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from trading_contracts import (
    ExecutionReportV2,
    InstrumentCapability,
    BrokerCapabilitySnapshot,
    OrderIntentV2,
    OwnerCommand,
    ProtectionLegState,
    ProtectiveBracket,
    RiskEnvelope,
    is_legal_transition,
    protection_covers_filled,
)

NOW = datetime(2026, 8, 2, 5, 0, tzinfo=timezone.utc)
CAP_HASH = "sha256:" + "c" * 64


def _identity(**kw):
    base = dict(
        object_id="oi-1",
        as_of=NOW,
        producer={"name": "lts.demo_execution", "version": "0.1.0"},
        trace_id="trace-1",
    )
    base.update(kw)
    return base


def _bracket(sl=0.98, tp=1.02, **kw):
    return ProtectiveBracket(stop_loss_price=sl, take_profit_price=tp, **kw)


def _risk(**kw):
    base = dict(
        risk_fraction_at_stop=0.005,
        gross_notional_fraction=0.05,
        margin_fraction=0.02,
        daily_loss_budget_fraction=0.02,
        reservation_id="rsv-1",
    )
    base.update(kw)
    return RiskEnvelope(**base)


def _entry(**kw):
    base = dict(
        **_identity(),
        account_ref="fp-account",
        asset_id="fx:USD/CAD",
        venue="ibkr_paper",
        instrument="USD.CAD",
        intent_class="risk_increasing",
        order_type="market",
        delta_units=1000.0,
        protection=_bracket(),
        risk=_risk(),
        capability_snapshot_hash=CAP_HASH,
        idempotency_key="idem-1",
    )
    base.update(kw)
    return OrderIntentV2(**base)


# ── Finding 039: naked entries are inexpressible ──

def test_naked_market_entry_rejects():
    with pytest.raises(ValidationError, match="stop loss"):
        _entry(protection=None)


def test_naked_limit_entry_rejects():
    with pytest.raises(ValidationError):
        _entry(order_type="limit", limit_price=1.0, protection=None)


def test_naked_stop_entry_rejects():
    with pytest.raises(ValidationError):
        _entry(order_type="stop", entry_trigger_price=1.01, protection=None)


def test_protected_market_entry_accepts_and_serializes_both_legs():
    intent = _entry()
    payload = intent.model_dump()
    assert payload["protection"]["stop_loss_price"] == 0.98
    assert payload["protection"]["take_profit_price"] == 1.02


def test_stop_entry_trigger_is_distinct_from_protective_stop():
    intent = _entry(
        order_type="stop",
        entry_trigger_price=1.01,
        protection=_bracket(sl=0.98, tp=1.05),
    )
    assert intent.entry_trigger_price != intent.protection.stop_loss_price


def test_trigger_equal_to_protective_stop_rejects_ambiguity():
    with pytest.raises(ValidationError, match="ambiguity"):
        _entry(
            order_type="stop",
            entry_trigger_price=0.99,
            protection=_bracket(sl=0.99, tp=1.05),
        )


def test_stop_entry_without_trigger_rejects():
    with pytest.raises(ValidationError, match="entry_trigger_price"):
        _entry(order_type="stop")


def test_market_entry_with_trigger_rejects():
    with pytest.raises(ValidationError):
        _entry(entry_trigger_price=1.01)


def test_close_only_action_needs_no_protection():
    intent = _entry(
        intent_class="risk_reducing",
        reduce_action="close",
        protection=None,
        risk=None,
        capability_snapshot_hash=None,
        delta_units=-1000.0,
    )
    assert intent.reduce_action == "close"


def test_risk_reducing_without_action_rejects():
    with pytest.raises(ValidationError, match="reduce_action"):
        _entry(intent_class="risk_reducing", protection=None, risk=None,
               capability_snapshot_hash=None)


def test_modify_protection_requires_bracket():
    with pytest.raises(ValidationError, match="ProtectiveBracket"):
        _entry(intent_class="risk_reducing", reduce_action="modify_protection",
               protection=None, risk=None, capability_snapshot_hash=None)


# ── Finding 039: side/price geometry and finite positive prices ──

def test_long_inverted_bracket_rejects():
    with pytest.raises(ValidationError, match="long entry requires"):
        _entry(protection=_bracket(sl=1.02, tp=0.98))


def test_short_inverted_bracket_rejects():
    with pytest.raises(ValidationError, match="short entry requires"):
        _entry(delta_units=-1000.0, protection=_bracket(sl=0.98, tp=1.02))


def test_short_correct_bracket_accepts():
    intent = _entry(delta_units=-1000.0, protection=_bracket(sl=1.02, tp=0.98))
    assert intent.delta_units < 0


def test_limit_entry_outside_bracket_rejects():
    with pytest.raises(ValidationError):
        _entry(order_type="limit", limit_price=1.05,
               protection=_bracket(sl=0.98, tp=1.02))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_non_finite_or_nonpositive_protection_prices_reject(bad):
    with pytest.raises(ValidationError):
        _entry(protection=_bracket(sl=bad, tp=1.02))


def test_zero_delta_risk_increasing_rejects():
    with pytest.raises(ValidationError, match="nonzero delta_units"):
        _entry(delta_units=0.0)


# ── Finding 040: risk dimensions are separate and mandatory ──

def test_missing_risk_envelope_rejects():
    with pytest.raises(ValidationError, match="RiskEnvelope"):
        _entry(risk=None)


def test_missing_capability_snapshot_rejects():
    with pytest.raises(ValidationError, match="capability_snapshot_hash"):
        _entry(capability_snapshot_hash=None)


@pytest.mark.parametrize("field", [
    "risk_fraction_at_stop", "gross_notional_fraction",
    "margin_fraction", "daily_loss_budget_fraction",
])
@pytest.mark.parametrize("bad", [0.0, -0.01, 1.5, float("nan")])
def test_risk_fractions_are_bounded_positive(field, bad):
    with pytest.raises(ValidationError):
        _risk(**{field: bad})


# ── Finding 041: lifecycle states, brackets, partial fills ──

def _report(**kw):
    base = dict(
        **_identity(object_id="er-1"),
        order_intent_id="oi-1",
        attempt_id="attempt-1",
        bracket_role="parent",
        state="accepted",
        requested_units=1000.0,
    )
    base.update(kw)
    return ExecutionReportV2(**base)


def test_partial_fill_state_exists_and_bounds_quantities():
    report = _report(state="partially_filled", filled_units=400.0,
                     previous_state="accepted")
    assert report.filled_units == 400.0


def test_partial_fill_with_full_quantity_rejects():
    with pytest.raises(ValidationError, match="partially_filled"):
        _report(state="partially_filled", filled_units=1000.0)


def test_filled_requires_exact_quantity():
    with pytest.raises(ValidationError, match="filled requires"):
        _report(state="filled", filled_units=999.0)


def test_overfill_rejects():
    with pytest.raises(ValidationError, match="exceed"):
        _report(filled_units=1001.0)


def test_orphan_protection_leg_requires_parent_identity():
    with pytest.raises(ValidationError, match="parent_order_intent_id"):
        _report(bracket_role="stop_loss")


def test_illegal_transition_rejects():
    with pytest.raises(ValidationError, match="illegal lifecycle transition"):
        _report(state="filled", previous_state="rejected", filled_units=1000.0)


def test_terminal_states_have_no_successors():
    for terminal in ("rejected", "cancelled", "expired", "closed"):
        assert not any(
            is_legal_transition(terminal, nxt)
            for nxt in ("accepted", "filled", "partially_filled")
        )


def test_unknown_state_forces_reconciliation_flag():
    with pytest.raises(ValidationError, match="reconciliation_required"):
        _report(state="unknown_requires_reconciliation")
    report = _report(state="unknown_requires_reconciliation",
                     reconciliation_required=True)
    assert report.reconciliation_required


def test_reconciliation_flag_without_unknown_state_rejects():
    with pytest.raises(ValidationError):
        _report(reconciliation_required=True)


def test_unconfirmed_protection_is_detected_as_uncovered():
    report = _report(
        state="filled", filled_units=1000.0, previous_state="accepted",
        protection_legs=[
            ProtectionLegState(leg="stop_loss", broker_confirmed=True,
                               covered_units=1000.0),
            ProtectionLegState(leg="take_profit", broker_confirmed=False,
                               covered_units=1000.0),
        ],
    )
    assert not protection_covers_filled(report)


def test_fully_confirmed_protection_covers():
    report = _report(
        state="filled", filled_units=1000.0, previous_state="accepted",
        protection_legs=[
            ProtectionLegState(leg="stop_loss", broker_confirmed=True,
                               covered_units=1000.0),
            ProtectionLegState(leg="take_profit", broker_confirmed=True,
                               covered_units=1000.0),
        ],
    )
    assert protection_covers_filled(report)


def test_partial_fill_with_partial_protection_coverage_detected():
    report = _report(
        state="partially_filled", filled_units=600.0, previous_state="accepted",
        protection_legs=[
            ProtectionLegState(leg="stop_loss", broker_confirmed=True,
                               covered_units=400.0),
            ProtectionLegState(leg="take_profit", broker_confirmed=True,
                               covered_units=600.0),
        ],
    )
    assert not protection_covers_filled(report)


# ── Finding 042: owner command is deterministic and risk-reducing only ──

def _command(**kw):
    base = dict(
        **_identity(object_id="cmd-1"),
        command="hold",
        issuer_id="owner-allowlisted",
        exact_phrase="HOLD ALL DEMO TRADING NOW",
        nonce="nonce-1",
        expires_at=NOW + timedelta(minutes=5),
        idempotency_key="cmd-idem-1",
    )
    base.update(kw)
    return OwnerCommand(**base)


def test_owner_command_verbs_are_risk_reducing_only():
    for verb in ("hold", "kill", "flatten_all", "cancel_pending"):
        assert _command(command=verb).command == verb
    with pytest.raises(ValidationError):
        _command(command="resume")
    with pytest.raises(ValidationError):
        _command(command="open_position")


def test_expired_command_window_rejects():
    with pytest.raises(ValidationError, match="expires_at"):
        _command(expires_at=NOW - timedelta(seconds=1))


def test_naive_expiry_rejects():
    with pytest.raises(ValidationError):
        _command(expires_at=datetime(2026, 8, 2, 6, 0))


# ── Capability snapshot (ruling R3: mandatory evidence provenance) ──

_CAP = dict(
    instrument="USD.CAD", tradeable=True, shortable=True, min_units=1.0,
    unit_step=1.0, price_decimals=5, native_stop_loss=True,
    native_take_profit=True, native_bracket=True,
)


def _snapshot(**kw):
    base = dict(
        **_identity(object_id="cap-1"),
        venue="ibkr_paper",
        account_fingerprint="fp-1",
        environment="paper",
        capability_evidence="live_observed",
        source_artifact_hash="sha256:" + "d" * 64,
        source_observed_at=NOW,
        instruments=[InstrumentCapability(**_CAP)],
    )
    base.update(kw)
    return BrokerCapabilitySnapshot(**base)


def test_capability_snapshot_rejects_duplicate_instruments():
    with pytest.raises(ValidationError, match="duplicate instrument"):
        _snapshot(instruments=[InstrumentCapability(**_CAP),
                               InstrumentCapability(**_CAP)])


def test_capability_evidence_class_is_mandatory():
    with pytest.raises(ValidationError):
        BrokerCapabilitySnapshot(
            **_identity(object_id="cap-2"),
            venue="ibkr_paper", account_fingerprint="fp-1",
            environment="paper",
            instruments=[InstrumentCapability(**_CAP)],
        )


def test_synthetic_fixture_requires_synthetic_fingerprint():
    with pytest.raises(ValidationError, match="synthetic"):
        _snapshot(capability_evidence="synthetic_fixture",
                  account_fingerprint="fp-real-looking")
    ok = _snapshot(capability_evidence="synthetic_fixture",
                   account_fingerprint="synthetic-ibkr-fixture-1")
    assert ok.capability_evidence == "synthetic_fixture"


def test_capability_evidence_rejects_unknown_class():
    with pytest.raises(ValidationError):
        _snapshot(capability_evidence="assumed_from_documentation")


# ── Finding 051: cancel/flatten must identify their target ──

def test_cancel_without_target_rejects():
    with pytest.raises(ValidationError, match="reduce_target_order_intent_id"):
        _entry(intent_class="risk_reducing", reduce_action="cancel",
               protection=None, risk=None, capability_snapshot_hash=None,
               delta_units=0.0)


def test_flatten_without_target_rejects():
    with pytest.raises(ValidationError, match="reduce_target_order_intent_id"):
        _entry(intent_class="risk_reducing", reduce_action="flatten",
               protection=None, risk=None, capability_snapshot_hash=None,
               delta_units=-10.0)


def test_cancel_with_target_and_broker_ids_accepts():
    intent = _entry(
        intent_class="risk_reducing", reduce_action="cancel",
        protection=None, risk=None, capability_snapshot_hash=None,
        delta_units=0.0,
        reduce_target_order_intent_id="oi2-rsv-abc",
        reduce_target_broker_ids={"ibkr_order_id": "17"},
    )
    assert intent.reduce_target_order_intent_id == "oi2-rsv-abc"


def test_close_still_needs_no_target():
    intent = _entry(intent_class="risk_reducing", reduce_action="close",
                    protection=None, risk=None, capability_snapshot_hash=None,
                    delta_units=-10.0)
    assert intent.reduce_target_order_intent_id is None


# ── Ruling R4: amended transition law ──

def test_r4_fill_before_ack_is_legal():
    assert is_legal_transition("requested", "filled")
    assert is_legal_transition("requested", "partially_filled")


def test_r4_cancel_before_ack_is_legal():
    assert is_legal_transition("requested", "cancel_pending")


def test_r4_expiry_while_cancel_pending_is_legal():
    assert is_legal_transition("cancel_pending", "expired")


def test_r4_reconciliation_paths_from_unknown():
    for target in ("cancel_pending", "modified",
                   "unknown_requires_reconciliation"):
        assert is_legal_transition("unknown_requires_reconciliation", target)


# ── v1 preservation: semantics migrate by version, never by edit ──

def test_v1_order_intent_unchanged_naked_entry_still_validates():
    from trading_contracts import OrderIntent

    naked = OrderIntent(
        **_identity(object_id="oi-v1"),
        account_ref="fp", asset_id="fx:USD/CAD", venue="v", instrument="i",
        order_type="market", delta_units=1.0, idempotency_key="k",
    )
    assert naked.stop_price is None and naked.take_profit_price is None
