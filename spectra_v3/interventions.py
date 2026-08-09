from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class ExpectedRelation(str, Enum):
    """Expected task-level effect of an intervention."""

    PRESERVE = "preserve"
    CHANGE = "change"
    CONDITIONAL = "conditional"


class VerificationStatus(str, Enum):
    """How strongly the intervention contract has been checked."""

    UNVERIFIED = "unverified"
    AUTOMATIC = "automatic"
    HUMAN = "human"


@dataclass(frozen=True)
class Intervention:
    base_text: str
    transformed_text: str
    family: str
    expected_relation: ExpectedRelation
    strength: float
    generator_id: str
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_text.strip():
            raise ValueError("base_text cannot be empty")
        if not self.transformed_text.strip():
            raise ValueError("transformed_text cannot be empty")
        if self.base_text == self.transformed_text:
            raise ValueError("an intervention must change the text")
        if not self.family.strip():
            raise ValueError("family cannot be empty")
        if not self.generator_id.strip():
            raise ValueError("generator_id cannot be empty")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_text": self.base_text,
            "transformed_text": self.transformed_text,
            "family": self.family,
            "expected_relation": self.expected_relation.value,
            "strength": self.strength,
            "generator_id": self.generator_id,
            "verification_status": self.verification_status.value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, base_text: str | None = None
    ) -> "Intervention":
        resolved_base = base_text if base_text is not None else str(value["base_text"])
        return cls(
            base_text=resolved_base,
            transformed_text=str(value["transformed_text"]),
            family=str(value["family"]),
            expected_relation=ExpectedRelation(value["expected_relation"]),
            strength=float(value.get("strength", 1.0)),
            generator_id=str(value["generator_id"]),
            verification_status=VerificationStatus(
                value.get("verification_status", VerificationStatus.UNVERIFIED.value)
            ),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class InterventionOrbit:
    base_id: str
    base_text: str
    interventions: tuple[Intervention, ...]
    source: str = ""
    context_text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_id.strip():
            raise ValueError("base_id cannot be empty")
        if not self.base_text.strip():
            raise ValueError("base_text cannot be empty")
        if not self.interventions:
            raise ValueError("an intervention orbit cannot be empty")
        if any(item.base_text != self.base_text for item in self.interventions):
            raise ValueError("all interventions must reference the orbit base_text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_id": self.base_id,
            "base_text": self.base_text,
            "source": self.source,
            "context_text": self.context_text,
            "metadata": dict(self.metadata),
            "interventions": [item.to_dict() for item in self.interventions],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InterventionOrbit":
        base_text = str(value["base_text"])
        interventions = tuple(
            Intervention.from_dict(item, base_text=base_text)
            for item in value["interventions"]
        )
        return cls(
            base_id=str(value["base_id"]),
            base_text=base_text,
            interventions=interventions,
            source=str(value.get("source", "")),
            context_text=str(value.get("context_text", "")),
            metadata=dict(value.get("metadata", {})),
        )


def make_exact_replacement(
    base_text: str,
    *,
    target: str,
    replacement: str,
    family: str,
    expected_relation: ExpectedRelation,
    generator_id: str,
    occurrence: int = 0,
    strength: float = 1.0,
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
    metadata: Mapping[str, Any] | None = None,
) -> Intervention:
    """Replace exactly one occurrence while retaining explicit semantic provenance."""

    if not target:
        raise ValueError("target cannot be empty")
    if not replacement:
        raise ValueError("replacement cannot be empty")
    if target == replacement:
        raise ValueError("target and replacement must differ")
    if occurrence < 0:
        raise ValueError("occurrence must be non-negative")

    starts: list[int] = []
    offset = 0
    while True:
        index = base_text.find(target, offset)
        if index < 0:
            break
        starts.append(index)
        offset = index + len(target)

    if occurrence >= len(starts):
        raise ValueError(
            f"target occurrence {occurrence} does not exist; found {len(starts)} occurrence(s)"
        )

    start = starts[occurrence]
    transformed = base_text[:start] + replacement + base_text[start + len(target) :]
    details = dict(metadata or {})
    details.update(
        {"target": target, "replacement": replacement, "occurrence": occurrence}
    )

    return Intervention(
        base_text=base_text,
        transformed_text=transformed,
        family=family,
        expected_relation=expected_relation,
        strength=strength,
        generator_id=generator_id,
        verification_status=verification_status,
        metadata=details,
    )


def make_negation_intervention(
    base_text: str,
    *,
    target: str,
    replacement: str,
    expected_relation: ExpectedRelation,
    generator_id: str,
    occurrence: int = 0,
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
) -> Intervention:
    return make_exact_replacement(
        base_text,
        target=target,
        replacement=replacement,
        family="negation",
        expected_relation=expected_relation,
        generator_id=generator_id,
        occurrence=occurrence,
        verification_status=verification_status,
    )


def make_entity_intervention(
    base_text: str,
    *,
    target: str,
    replacement: str,
    expected_relation: ExpectedRelation,
    generator_id: str,
    occurrence: int = 0,
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
) -> Intervention:
    return make_exact_replacement(
        base_text,
        target=target,
        replacement=replacement,
        family="entity",
        expected_relation=expected_relation,
        generator_id=generator_id,
        occurrence=occurrence,
        verification_status=verification_status,
    )


def make_quantity_intervention(
    base_text: str,
    *,
    target: str,
    replacement: str,
    expected_relation: ExpectedRelation,
    generator_id: str,
    occurrence: int = 0,
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
) -> Intervention:
    return make_exact_replacement(
        base_text,
        target=target,
        replacement=replacement,
        family="quantity",
        expected_relation=expected_relation,
        generator_id=generator_id,
        occurrence=occurrence,
        verification_status=verification_status,
    )


def make_surface_intervention(
    base_text: str,
    *,
    transformed_text: str,
    generator_id: str,
    strength: float = 0.1,
    verification_status: VerificationStatus = VerificationStatus.AUTOMATIC,
    metadata: Mapping[str, Any] | None = None,
) -> Intervention:
    return Intervention(
        base_text=base_text,
        transformed_text=transformed_text,
        family="surface",
        expected_relation=ExpectedRelation.PRESERVE,
        strength=strength,
        generator_id=generator_id,
        verification_status=verification_status,
        metadata=dict(metadata or {}),
    )


def validate_unique_generator_ids(interventions: Iterable[Intervention]) -> None:
    ids = [item.generator_id for item in interventions]
    if len(ids) != len(set(ids)):
        raise ValueError("generator_id values must be unique within an orbit")
