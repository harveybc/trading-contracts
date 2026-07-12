from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from trading_contracts import (
    ComponentManifest,
    ContractEdge,
    DeploymentManifest,
    contract_family_major,
    contract_versions_compatible,
    evaluate_deployment_compatibility,
)


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _bundle() -> tuple[ComponentManifest, DeploymentManifest]:
    component = ComponentManifest.model_validate_json(
        (EXAMPLES / "component_manifest_asset_policy.json").read_text(encoding="utf-8")
    )
    deployment = DeploymentManifest.model_validate_json(
        (EXAMPLES / "deployment_manifest_smoke.json").read_text(encoding="utf-8")
    )
    return component, deployment


def test_contract_major_compatibility() -> None:
    assert contract_family_major("asset_intent.v1") == ("asset_intent", 1)
    assert contract_versions_compatible("asset_intent.v1", "asset_intent.v1")
    assert not contract_versions_compatible("asset_intent.v1", "asset_intent.v2")
    assert not contract_versions_compatible("asset_intent.v1", "order_intent.v1")


def test_generated_example_bundle_is_compatible() -> None:
    component, deployment = _bundle()
    report = evaluate_deployment_compatibility(
        deployment,
        {"asset_policy": component},
        required_roles=["asset_policy"],
    )
    assert report.compatible is True
    assert report.issues == []


def test_hash_and_cutoff_mismatches_fail() -> None:
    component, deployment = _bundle()
    changed_component = component.model_copy(
        update={"training_cutoff": deployment.training_cutoff + timedelta(seconds=1)}
    )
    report = evaluate_deployment_compatibility(
        deployment,
        {"asset_policy": changed_component},
        required_roles=["asset_policy"],
    )
    codes = {issue.code for issue in report.issues}
    assert report.compatible is False
    assert "future_training_cutoff" in codes
    assert "component_hash_mismatch" in codes


def test_contract_edge_checks_both_sides() -> None:
    component, deployment = _bundle()
    consumer = component.model_copy(
        update={
            "object_id": "consumer",
            "role": "portfolio_allocator",
            "input_contract_versions": ["portfolio_intent.v1"],
            "output_contract_versions": ["portfolio_intent.v1"],
        }
    )
    deployment_payload = deployment.model_dump(mode="python")
    from trading_contracts import content_hash

    deployment_payload["components"]["portfolio_allocator"] = content_hash(consumer)
    updated_deployment = DeploymentManifest.model_validate(deployment_payload)
    report = evaluate_deployment_compatibility(
        updated_deployment,
        {"asset_policy": component, "portfolio_allocator": consumer},
        contract_edges=[
            ContractEdge(
                producer_role="asset_policy",
                consumer_role="portfolio_allocator",
                contract_version="asset_intent.v1",
            )
        ],
    )
    assert report.compatible is False
    assert {issue.code for issue in report.issues} == {"consumer_contract_missing"}


def test_missing_required_role_fails() -> None:
    component, deployment = _bundle()
    report = evaluate_deployment_compatibility(
        deployment,
        {"asset_policy": component},
        required_roles=["asset_policy", "portfolio_allocator"],
    )
    assert report.compatible is False
    assert any(issue.code == "missing_required_role" for issue in report.issues)


def test_malformed_execution_version_fails_closed() -> None:
    component, deployment = _bundle()
    payload = deployment.model_dump(mode="python")
    payload["compatibility"]["execution_contract_version"] = "unversioned"
    malformed = DeploymentManifest.model_validate(payload)
    report = evaluate_deployment_compatibility(
        malformed,
        {"asset_policy": component},
    )
    assert report.compatible is False
    assert {issue.code for issue in report.issues} == {"execution_contract_mismatch"}
