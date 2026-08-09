from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .lexical_variables import select_relevant_span


FRAME_COORDINATE_NAMES = (
    "predicate_alignment",
    "argument_alignment",
    "direction_alignment",
    "scope_alignment",
    "modality_alignment",
    "reliability",
)

_ENTITY_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*)+\b"
)
_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NEGATIONS = {"not", "never", "neither", "nor"}
_POSSIBILITY = {"may", "might", "could", "possibly", "perhaps"}
_CERTAINTY = {"certainly", "definitely", "clearly", "confirmedly"}
_AUXILIARIES = {
    "be",
    "been",
    "being",
    "did",
    "does",
    "had",
    "has",
    "have",
    "is",
    "was",
    "were",
    *_NEGATIONS,
    *_POSSIBILITY,
    *_CERTAINTY,
}


def _words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]


def _relation_phrase(text: str, left_end: int, right_start: int) -> str:
    phrase = text[left_end:right_start]
    words = [word for word in _words(phrase) if word != "by"]
    predicate = [word for word in words if word not in _AUXILIARIES]
    return " ".join(predicate)


@dataclass(frozen=True)
class SemanticFrame:
    """A small query-conditioned predicate/argument observation.

    The extractor intentionally has no access to intervention family, label, or
    generator metadata. It is a structural development observer, not a general
    semantic parser.
    """

    span: str
    predicate: str
    actor: str
    patient: str
    polarity: str
    modality: str
    voice: str
    reliability: float

    def __post_init__(self) -> None:
        if not self.span.strip():
            raise ValueError("frame span cannot be empty")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("frame reliability must be between zero and one")

    def to_dict(self) -> dict[str, str | float]:
        return {
            "span": self.span,
            "predicate": self.predicate,
            "actor": self.actor,
            "patient": self.patient,
            "polarity": self.polarity,
            "modality": self.modality,
            "voice": self.voice,
            "reliability": float(self.reliability),
        }


def extract_semantic_frame(query: str, document: str) -> SemanticFrame:
    """Extract an ordered binary frame from the query-relevant sentence.

    This deliberately narrow parser supports active and passive binary clauses.
    Reliability exposes when that structural assumption does not hold, allowing a
    later cascade to abstain instead of silently inventing a frame.
    """

    if not query.strip() or not document.strip():
        raise ValueError("query and document cannot be empty")
    span = select_relevant_span(query, document)
    mentions = list(_ENTITY_PATTERN.finditer(span))
    if len(mentions) < 2:
        return SemanticFrame(
            span=span,
            predicate="",
            actor="",
            patient="",
            polarity="unknown",
            modality="unknown",
            voice="unknown",
            reliability=0.0,
        )

    left, right = mentions[0], mentions[1]
    between = span[left.end() : right.start()]
    between_words = set(_words(between))
    passive = "by" in between_words and bool(
        between_words & {"is", "was", "were", "been", "being"}
    )
    if passive:
        actor, patient = right.group(0), left.group(0)
        voice = "passive"
    else:
        actor, patient = left.group(0), right.group(0)
        voice = "active"

    predicate = _relation_phrase(span, left.end(), right.start())
    polarity = "negated" if between_words & _NEGATIONS else "affirmed"
    modality = "possible" if between_words & _POSSIBILITY else "asserted"
    reliability = 1.0 if predicate else 0.5
    if len(mentions) > 2:
        reliability *= 0.75
    return SemanticFrame(
        span=span,
        predicate=predicate,
        actor=actor,
        patient=patient,
        polarity=polarity,
        modality=modality,
        voice=voice,
        reliability=reliability,
    )


@dataclass(frozen=True)
class FrameCompatibility:
    predicate_alignment: float
    argument_alignment: float
    direction_alignment: float
    scope_alignment: float
    modality_alignment: float
    reliability: float

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name)) for name in FRAME_COORDINATE_NAMES
        }

    def drops_from(self, base: "FrameCompatibility") -> dict[str, float]:
        return {
            f"{name}_drop": max(
                float(getattr(base, name) - getattr(self, name)), 0.0
            )
            for name in FRAME_COORDINATE_NAMES[:-1]
        }


def frame_compatibility(
    query: SemanticFrame,
    candidate: SemanticFrame,
    similarity: Callable[[str, str], float],
) -> FrameCompatibility:
    """Compare semantic slots while keeping unordered arguments and role order apart."""

    if not query.predicate or not candidate.predicate:
        return FrameCompatibility(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    predicate = similarity(query.predicate, candidate.predicate)
    direct = 0.5 * (
        similarity(query.actor, candidate.actor)
        + similarity(query.patient, candidate.patient)
    )
    swapped = 0.5 * (
        similarity(query.actor, candidate.patient)
        + similarity(query.patient, candidate.actor)
    )
    return FrameCompatibility(
        predicate_alignment=predicate,
        argument_alignment=max(direct, swapped),
        direction_alignment=direct,
        scope_alignment=float(query.polarity == candidate.polarity),
        modality_alignment=float(query.modality == candidate.modality),
        reliability=min(query.reliability, candidate.reliability),
    )


def fixed_frame_distance(drops: dict[str, float]) -> float:
    """Fixed monotone norm; the predicate cosine is mapped from range two to one."""

    required = {
        "predicate_alignment_drop",
        "argument_alignment_drop",
        "direction_alignment_drop",
        "scope_alignment_drop",
        "modality_alignment_drop",
    }
    missing = required - set(drops)
    if missing:
        raise ValueError(f"missing frame drops: {sorted(missing)}")
    return max(
        0.5 * max(float(drops["predicate_alignment_drop"]), 0.0),
        max(float(drops["argument_alignment_drop"]), 0.0),
        max(float(drops["direction_alignment_drop"]), 0.0),
        max(float(drops["scope_alignment_drop"]), 0.0),
        max(float(drops["modality_alignment_drop"]), 0.0),
    )
