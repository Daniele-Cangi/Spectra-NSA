from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import log
from typing import Any, Mapping

from .interventions import ExpectedRelation


class SemanticAxis(str, Enum):
    """Controlled coordinates of a task-relative semantic state."""

    SURFACE = "surface"
    ENTITY = "entity"
    RELATION = "relation"
    QUANTITY = "quantity"
    TIME = "time"
    POLARITY = "polarity"
    MODALITY = "modality"
    SCOPE = "scope"


@dataclass(frozen=True)
class SemanticVariableChange:
    """Oracle annotation for one controlled semantic variable change.

    This annotation belongs to the experimental design. It must never be fed to a
    learned detector as an input feature.
    """

    axis: SemanticAxis
    frame_id: str
    query_relevant: bool
    value_changed: bool
    before_value: str
    after_value: str

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame_id cannot be empty")
        if not self.before_value.strip() or not self.after_value.strip():
            raise ValueError("semantic values cannot be empty")
        if self.value_changed and self.before_value == self.after_value:
            raise ValueError("a changed semantic value needs distinct endpoints")
        if not self.value_changed and self.before_value != self.after_value:
            raise ValueError("an unchanged semantic value needs identical endpoints")

    @property
    def expected_relation(self) -> ExpectedRelation:
        if self.query_relevant and self.value_changed:
            return ExpectedRelation.CHANGE
        return ExpectedRelation.PRESERVE

    def validate_relation(self, relation: ExpectedRelation) -> None:
        if relation != self.expected_relation:
            raise ValueError(
                "semantic variable contract disagrees with expected_relation: "
                f"expected {self.expected_relation.value}, received {relation.value}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis.value,
            "frame_id": self.frame_id,
            "query_relevant": self.query_relevant,
            "value_changed": self.value_changed,
            "before_value": self.before_value,
            "after_value": self.after_value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticVariableChange":
        return cls(
            axis=SemanticAxis(str(value["axis"])),
            frame_id=str(value["frame_id"]),
            query_relevant=bool(value["query_relevant"]),
            value_changed=bool(value["value_changed"]),
            before_value=str(value["before_value"]),
            after_value=str(value["after_value"]),
        )


RELATIONAL_STATE_NAMES = (
    "support",
    "contradiction",
    "neutrality",
    "commitment",
    "entropy",
)


@dataclass(frozen=True)
class RelationalState:
    """Task-conditioned observable state derived from an NLI distribution."""

    support: float
    contradiction: float
    neutrality: float
    commitment: float
    entropy: float

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name)) for name in RELATIONAL_STATE_NAMES
        }

    def delta_from(self, base: "RelationalState") -> dict[str, float]:
        return {
            f"{name}_delta": float(getattr(self, name) - getattr(base, name))
            for name in RELATIONAL_STATE_NAMES
        }


def _unique_probability(
    probabilities: Mapping[str, float], label_fragment: str
) -> float:
    matches = [
        float(probability)
        for label, probability in probabilities.items()
        if label_fragment in str(label).casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cannot identify one {label_fragment!r} probability: {dict(probabilities)}"
        )
    return matches[0]


def relational_state_from_probabilities(
    probabilities: Mapping[str, float],
) -> RelationalState:
    """Convert labelled NLI probabilities into interpretable relation variables."""

    if len(probabilities) < 2:
        raise ValueError("at least two labelled probabilities are required")
    raw = {str(label): float(value) for label, value in probabilities.items()}
    if any(value < 0.0 for value in raw.values()):
        raise ValueError("probabilities cannot be negative")
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    normalized = {label: value / total for label, value in raw.items()}
    entailment = _unique_probability(normalized, "entail")
    contradiction = _unique_probability(normalized, "contradict")
    neutrality = _unique_probability(normalized, "neutral")
    entropy = -sum(
        value * log(value) for value in normalized.values() if value > 0.0
    ) / log(len(normalized))
    return RelationalState(
        support=entailment - contradiction,
        contradiction=contradiction,
        neutrality=neutrality,
        commitment=entailment + contradiction,
        entropy=entropy,
    )


def semantic_change_from_metadata(
    metadata: Mapping[str, Any],
) -> SemanticVariableChange:
    value = metadata.get("semantic_change")
    if not isinstance(value, Mapping):
        raise ValueError("intervention metadata has no semantic_change annotation")
    return SemanticVariableChange.from_dict(value)
