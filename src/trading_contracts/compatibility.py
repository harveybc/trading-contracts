from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import Field

from .canonical import content_hash
from .contracts import ComponentManifest, DeploymentManifest, NonEmpty, StrictModel


_VERSION = re.compile(r"^(?P<family>[a-z][a-z0-9_.-]*)\.v(?P<major>[1-9][0-9]*)$")


class ContractEdge(StrictModel):
    producer_role: NonEmpty
    consumer_role: NonEmpty
    contract_version: NonEmpty


class CompatibilityIssue(StrictModel):
    code: NonEmpty
    message: NonEmpty
    roles: list[str] = Field(default_factory=list)


class CompatibilityReport(StrictModel):
    compatible: bool
    issues: list[CompatibilityIssue]
    component_manifest_hashes: dict[str, str]


def contract_family_major(version: str) -> tuple[str, int]:
    match = _VERSION.fullmatch(version)
    if not match:
        raise ValueError(f"invalid contract version: {version!r}")
    return match.group("family"), int(match.group("major"))


def contract_versions_compatible(required: str, available: str) -> bool:
    return contract_family_major(required) == contract_family_major(available)


def _has_compatible(versions: Sequence[str], required: str) -> bool:
    return any(contract_versions_compatible(required, version) for version in versions)


def _safe_compatible(required: object, available: object) -> bool:
    if not isinstance(required, str) or not isinstance(available, str):
        return False
    try:
        return contract_versions_compatible(required, available)
    except ValueError:
        return False


def _safe_has_compatible(versions: Sequence[str], required: str) -> bool:
    try:
        return _has_compatible(versions, required)
    except ValueError:
        return False


def evaluate_deployment_compatibility(
    deployment: DeploymentManifest,
    components: Mapping[str, ComponentManifest],
    *,
    required_roles: Sequence[str] = (),
    contract_edges: Sequence[ContractEdge] = (),
) -> CompatibilityReport:
    issues: list[CompatibilityIssue] = []
    hashes = {role: content_hash(component) for role, component in components.items()}

    for role in sorted(set(required_roles)):
        if role not in components:
            issues.append(
                CompatibilityIssue(
                    code="missing_required_role",
                    message=f"required component role is missing: {role}",
                    roles=[role],
                )
            )

    for role, component in sorted(components.items()):
        if component.role != role:
            issues.append(
                CompatibilityIssue(
                    code="role_key_mismatch",
                    message=(
                        f"component map key {role!r} does not match manifest role "
                        f"{component.role!r}"
                    ),
                    roles=[role, component.role],
                )
            )
        if component.training_cutoff > deployment.training_cutoff:
            issues.append(
                CompatibilityIssue(
                    code="future_training_cutoff",
                    message=(
                        f"component {role!r} cutoff {component.training_cutoff.isoformat()} "
                        f"exceeds deployment cutoff {deployment.training_cutoff.isoformat()}"
                    ),
                    roles=[role],
                )
            )
        bound_hash = deployment.components.get(role)
        if bound_hash is None:
            issues.append(
                CompatibilityIssue(
                    code="unbound_component",
                    message=f"deployment does not bind component role: {role}",
                    roles=[role],
                )
            )
        elif bound_hash != hashes[role]:
            issues.append(
                CompatibilityIssue(
                    code="component_hash_mismatch",
                    message=f"deployment hash does not match component manifest for role: {role}",
                    roles=[role],
                )
            )

        required_execution = deployment.compatibility.get(
            "execution_contract_version",
            component.execution_contract_version,
        )
        if not _safe_compatible(required_execution, component.execution_contract_version):
            issues.append(
                CompatibilityIssue(
                    code="execution_contract_mismatch",
                    message=f"execution contract is incompatible for role: {role}",
                    roles=[role],
                )
            )

    for role in sorted(set(deployment.components) - set(components)):
        issues.append(
            CompatibilityIssue(
                code="missing_bound_manifest",
                message=f"deployment binds a manifest that was not supplied: {role}",
                roles=[role],
            )
        )

    for edge in contract_edges:
        producer = components.get(edge.producer_role)
        consumer = components.get(edge.consumer_role)
        if producer is None or consumer is None:
            continue
        if not _safe_has_compatible(producer.output_contract_versions, edge.contract_version):
            issues.append(
                CompatibilityIssue(
                    code="producer_contract_missing",
                    message=(
                        f"producer {edge.producer_role!r} does not emit "
                        f"{edge.contract_version!r}"
                    ),
                    roles=[edge.producer_role, edge.consumer_role],
                )
            )
        if not _safe_has_compatible(consumer.input_contract_versions, edge.contract_version):
            issues.append(
                CompatibilityIssue(
                    code="consumer_contract_missing",
                    message=(
                        f"consumer {edge.consumer_role!r} does not accept "
                        f"{edge.contract_version!r}"
                    ),
                    roles=[edge.producer_role, edge.consumer_role],
                )
            )

    return CompatibilityReport(
        compatible=not issues,
        issues=issues,
        component_manifest_hashes=hashes,
    )
