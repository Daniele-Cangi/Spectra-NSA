from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

LEXICAL_STATE_NAMES = (
    "content_coverage",
    "entity_coverage",
    "numeric_coverage",
    "polarity_alignment",
    "scope_overlap",
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:,\d+)*")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_NEGATIONS = {"no", "not", "never", "neither", "nor", "without"}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "which",
}


def _matches(text: str) -> list[re.Match[str]]:
    return list(_TOKEN_PATTERN.finditer(text))


def _tokens(text: str) -> list[str]:
    return [
        token.replace(",", "") if token[0].isdigit() else token.casefold()
        for match in _matches(text)
        for token in (match.group(0),)
    ]


def _content_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in _STOPWORDS}


def _entity_tokens(text: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in _matches(text)
        if match.group(0)[0].isupper()
        and match.group(0).casefold() not in _STOPWORDS
    }


def _numeric_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if token[0].isdigit()}


def _coverage(expected: set[str], observed: set[str]) -> float:
    if not expected:
        return 1.0
    return len(expected & observed) / len(expected)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def select_relevant_span(query: str, document: str) -> str:
    """Select the sentence with maximum query-content coverage."""

    query_tokens = _content_tokens(query)
    spans = [span.strip() for span in _SENTENCE_PATTERN.split(document) if span.strip()]
    if not spans:
        raise ValueError("document contains no non-empty span")
    return max(
        spans,
        key=lambda span: (
            _coverage(query_tokens, _content_tokens(span)),
            _jaccard(query_tokens, _content_tokens(span)),
        ),
    )


@dataclass(frozen=True)
class LexicalVariableState:
    content_coverage: float
    entity_coverage: float
    numeric_coverage: float
    polarity_alignment: float
    scope_overlap: float

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name)) for name in LEXICAL_STATE_NAMES
        }

    def delta_from(self, base: "LexicalVariableState") -> dict[str, float]:
        return {
            f"{name}_delta": float(getattr(self, name) - getattr(base, name))
            for name in LEXICAL_STATE_NAMES
        }


def lexical_variable_state(query: str, document: str) -> LexicalVariableState:
    if not query.strip() or not document.strip():
        raise ValueError("query and document cannot be empty")
    span = select_relevant_span(query, document)
    query_content = _content_tokens(query)
    span_content = _content_tokens(span)
    query_negated = bool(set(_tokens(query)) & _NEGATIONS)
    span_negated = bool(set(_tokens(span)) & _NEGATIONS)
    return LexicalVariableState(
        content_coverage=_coverage(query_content, span_content),
        entity_coverage=_coverage(_entity_tokens(query), _entity_tokens(span)),
        numeric_coverage=_coverage(_numeric_tokens(query), _numeric_tokens(span)),
        polarity_alignment=float(query_negated == span_negated),
        scope_overlap=_jaccard(query_content, span_content),
    )


def lexical_state_distance(
    base: LexicalVariableState, transformed: LexicalVariableState
) -> float:
    """Family-blind L-infinity distance over equally routed state coordinates."""

    delta = np.asarray(
        [getattr(transformed, name) - getattr(base, name) for name in LEXICAL_STATE_NAMES],
        dtype=np.float64,
    )
    return float(np.max(np.abs(delta)))
