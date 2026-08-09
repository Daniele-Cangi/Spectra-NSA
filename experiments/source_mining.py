from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


SCHEMA_VERSION = 1
SUPPORTED_AXES = frozenset({"relation", "direction", "scope", "modality"})
SUPPORTED_MODES = frozenset({"seed-only", "auto-proposal"})
ASSISTANCE_LEVELS = frozenset(
    {"source-only", "query-frame-axis", "full-triplet"}
)
ROLES = ("critical", "control", "invariant")
STACKEXCHANGE_API = "https://api.stackexchange.com/2.3/questions"
_PLACEHOLDER_PREFIX = "<human-"
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)")
_URL = re.compile(r"https?://\S+", re.I)
_LEAKAGE = re.compile(
    r"\b(?:spectra|auroc|expected_relation|frame observer|nli fallback|model output)\b",
    re.I,
)
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")
_AXIS_CONTRACTS = {
    "relation": (
        "Change the query-relevant event or predicate in the critical item; "
        "apply the same kind of change only to an irrelevant proposition in the "
        "control; preserve the relevant relation in a fluent invariant rewrite."
    ),
    "direction": (
        "Reverse or reassign actor/patient direction only in the critical item; "
        "perform a matched direction change on an irrelevant proposition in the "
        "control; preserve actor/patient roles in the invariant."
    ),
    "scope": (
        "Change the scope of negation or a quantifier only for the relevant "
        "proposition; make a matched irrelevant-scope control; preserve the "
        "relevant scope in the invariant."
    ),
    "modality": (
        "Change permission, obligation, possibility, or certainty only for the "
        "relevant proposition; make a matched irrelevant-modality control; "
        "preserve the relevant modality in the invariant."
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"input contains no records: {path}")
    return rows


def _require_text(value: Any, field: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{field} cannot be empty")
    return resolved


def _slug(value: str, *, prefix: str = "") -> str:
    resolved = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.").lower()
    if not resolved:
        resolved = "unknown"
    return f"{prefix}{resolved}"[:95]


def _canonical_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", "")
    )


def _timestamp(value: int | float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()


def _parse_date(value: str | None, *, end: bool = False) -> int | None:
    if value is None:
        return None
    parsed = date.fromisoformat(value)
    clock = datetime_time.max if end else datetime_time.min
    return int(datetime.combine(parsed, clock, tzinfo=timezone.utc).timestamp())


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag in {"pre", "code", "blockquote"}:
            self.skip_depth += 1
        elif self.skip_depth == 0 and tag in {"p", "br", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"pre", "code", "blockquote"} and self.skip_depth:
            self.skip_depth -= 1
        elif self.skip_depth == 0 and tag in {"p", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self.parts.append(data)


def _normalize_text(value: str) -> str:
    value = html.unescape(value).replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def html_to_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return _normalize_text("".join(parser.parts))


def _redact_unnecessary_pii(value: str) -> tuple[str, tuple[str, ...]]:
    redactions = []
    for label, pattern in (("email", _EMAIL), ("phone", _PHONE), ("url", _URL)):
        if pattern.search(value):
            redactions.append(label)
            value = pattern.sub(f"[redacted-{label}]", value)
    return value, tuple(redactions)


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    platform: str
    site: str
    item_id: str
    canonical_url: str
    retrieved_utc: str
    created_utc: str | None
    content_license: str
    source_type: str
    source_hash: str
    attribution_display_name: str
    attribution_url: str | None
    title: str
    text: str
    tags: tuple[str, ...]
    redactions: tuple[str, ...] = ()

    def validate(self) -> None:
        for field in (
            "source_id",
            "platform",
            "site",
            "item_id",
            "canonical_url",
            "retrieved_utc",
            "content_license",
            "source_type",
            "source_hash",
            "attribution_display_name",
            "title",
            "text",
        ):
            _require_text(getattr(self, field), field)
        if self.platform != "stackexchange":
            raise ValueError("unsupported source platform")
        parsed_url = urllib.parse.urlsplit(self.canonical_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("canonical_url must be an absolute HTTPS URL")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_hash):
            raise ValueError("source_hash must be a SHA-256 digest")
        if _sha256_text(self.text) != self.source_hash:
            raise ValueError("source_hash does not match normalized source text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_id": self.source_id,
            "platform": self.platform,
            "site": self.site,
            "item_id": self.item_id,
            "canonical_url": self.canonical_url,
            "retrieved_utc": self.retrieved_utc,
            "created_utc": self.created_utc,
            "content_license": self.content_license,
            "source_type": self.source_type,
            "source_hash": self.source_hash,
            "attribution": {
                "display_name": self.attribution_display_name,
                "profile_url": self.attribution_url,
            },
            "title": self.title,
            "text": self.text,
            "tags": list(self.tags),
            "redactions": list(self.redactions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceRecord":
        if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported source schema version")
        attribution = value.get("attribution")
        if not isinstance(attribution, Mapping):
            raise ValueError("source attribution must be an object")
        record = cls(
            source_id=_require_text(value["source_id"], "source_id"),
            platform=_require_text(value["platform"], "platform"),
            site=_require_text(value["site"], "site"),
            item_id=_require_text(value["item_id"], "item_id"),
            canonical_url=_require_text(value["canonical_url"], "canonical_url"),
            retrieved_utc=_require_text(value["retrieved_utc"], "retrieved_utc"),
            created_utc=(
                str(value["created_utc"]) if value.get("created_utc") else None
            ),
            content_license=_require_text(
                value["content_license"], "content_license"
            ),
            source_type=_require_text(value["source_type"], "source_type"),
            source_hash=_require_text(value["source_hash"], "source_hash"),
            attribution_display_name=_require_text(
                attribution["display_name"], "attribution.display_name"
            ),
            attribution_url=(
                str(attribution["profile_url"])
                if attribution.get("profile_url")
                else None
            ),
            title=_require_text(value["title"], "title"),
            text=_require_text(value["text"], "text"),
            tags=tuple(map(str, value.get("tags", []))),
            redactions=tuple(map(str, value.get("redactions", []))),
        )
        record.validate()
        return record


JsonTransport = Callable[[str], Mapping[str, Any]]


def _http_json(url: str) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "User-Agent": "Spectra-NSA-source-miner/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            payload = gzip.decompress(payload)
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Stack Exchange API response is not an object")
    return value


class StackExchangeAdapter:
    def __init__(
        self,
        *,
        transport: JsonTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport or _http_json
        self._sleep = sleep

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "adapter": "stackexchange-questions",
            "api_version": "2.3",
            "endpoint": STACKEXCHANGE_API,
            "filter": "withbody",
            "html_policy": "drop-code-pre-blockquote-v1",
            "pii_policy": "redact-before-persist-v1",
        }

    def fetch(
        self,
        *,
        site: str,
        from_date: int | None,
        to_date: int | None,
        max_items: int,
    ) -> tuple[list[SourceRecord], dict[str, Any]]:
        if max_items < 1 or max_items > 100:
            raise ValueError("Stack Exchange max_items must be between 1 and 100")
        parameters: dict[str, Any] = {
            "site": site,
            "pagesize": max_items,
            "page": 1,
            "order": "desc",
            "sort": "votes",
            "filter": "withbody",
        }
        if from_date is not None:
            parameters["fromdate"] = from_date
        if to_date is not None:
            parameters["todate"] = to_date
        url = f"{STACKEXCHANGE_API}?{urllib.parse.urlencode(parameters)}"
        payload = self._transport(url)
        if payload.get("error_id"):
            raise RuntimeError(
                f"Stack Exchange API error {payload.get('error_name')}: "
                f"{payload.get('error_message')}"
            )
        backoff = int(payload.get("backoff", 0))
        if backoff:
            self._sleep(float(backoff))
        retrieved = _utc_now()
        records = []
        for item in payload.get("items", []):
            if not isinstance(item, Mapping):
                continue
            body = html_to_text(str(item.get("body", "")))
            body, redactions = _redact_unnecessary_pii(body)
            owner = item.get("owner")
            if not isinstance(owner, Mapping):
                owner = {}
            item_id = str(item.get("question_id", ""))
            title = _normalize_text(html.unescape(str(item.get("title", ""))))
            title, title_redactions = _redact_unnecessary_pii(title)
            redactions = tuple(sorted(set((*redactions, *title_redactions))))
            link = _canonical_url(str(item.get("link", "")))
            if not body or not item_id or not title or not link:
                continue
            record = SourceRecord(
                source_id=f"stackexchange:{site}:question:{item_id}",
                platform="stackexchange",
                site=site,
                item_id=item_id,
                canonical_url=link,
                retrieved_utc=retrieved,
                created_utc=_timestamp(item.get("creation_date")),
                content_license=str(item.get("content_license", "unknown")),
                source_type="question",
                source_hash=_sha256_text(body),
                attribution_display_name=str(
                    owner.get("display_name", "community-user")
                ),
                attribution_url=(
                    _canonical_url(str(owner["link"])) if owner.get("link") else None
                ),
                title=title,
                text=body,
                tags=tuple(map(str, item.get("tags", []))),
                redactions=redactions,
            )
            record.validate()
            records.append(record)
        return records, {
            "request_url": url,
            "quota_remaining": payload.get("quota_remaining"),
            "quota_max": payload.get("quota_max"),
            "backoff_seconds": backoff,
            "has_more": bool(payload.get("has_more", False)),
        }


@dataclass(frozen=True)
class InferenceProposal:
    proposal_id: str
    source_id: str
    minimal_span: str
    surrounding_context: str
    query: str
    main_proposition: str
    actor: str
    predicate: str
    patient: str | None
    polarity: str
    modality: str
    scope_cues: tuple[str, ...]
    axis_candidates: tuple[str, ...]
    primary_axis: str
    confidence: float
    ambiguity_flags: tuple[str, ...]
    rejection_reason: str | None
    predicate_family: str
    syntactic_form: str
    source_domain: str
    mode: str
    assistance_level: str
    transformations: Mapping[str, Mapping[str, Any]] | None = None
    verifier_agreement: bool | None = None

    def validate(self) -> None:
        for field in (
            "proposal_id",
            "source_id",
            "minimal_span",
            "surrounding_context",
            "query",
            "main_proposition",
            "actor",
            "predicate",
            "polarity",
            "modality",
            "primary_axis",
            "predicate_family",
            "syntactic_form",
            "source_domain",
            "mode",
            "assistance_level",
        ):
            _require_text(getattr(self, field), field)
        if self.primary_axis not in SUPPORTED_AXES:
            raise ValueError("primary_axis is unsupported")
        if not set(self.axis_candidates) <= SUPPORTED_AXES:
            raise ValueError("axis_candidates contains an unsupported axis")
        if self.primary_axis not in self.axis_candidates:
            raise ValueError("primary_axis must appear in axis_candidates")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.mode not in SUPPORTED_MODES:
            raise ValueError("unsupported inference mode")
        if self.assistance_level not in ASSISTANCE_LEVELS:
            raise ValueError("unsupported assistance level")
        if self.mode == "seed-only" and self.transformations is not None:
            raise ValueError("seed-only inference cannot contain transformations")
        if self.mode == "auto-proposal":
            if self.assistance_level != "full-triplet":
                raise ValueError("auto-proposal requires full-triplet assistance")
            if self.transformations is None or set(self.transformations) != set(ROLES):
                raise ValueError("auto-proposal requires all three roles")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "source_id": self.source_id,
            "minimal_span": self.minimal_span,
            "surrounding_context": self.surrounding_context,
            "query": self.query,
            "main_proposition": self.main_proposition,
            "frame": {
                "actor": self.actor,
                "predicate": self.predicate,
                "patient": self.patient,
                "polarity": self.polarity,
                "modality": self.modality,
                "scope_cues": list(self.scope_cues),
            },
            "axis_candidates": list(self.axis_candidates),
            "primary_axis": self.primary_axis,
            "confidence": self.confidence,
            "ambiguity_flags": list(self.ambiguity_flags),
            "rejection_reason": self.rejection_reason,
            "predicate_family": self.predicate_family,
            "syntactic_form": self.syntactic_form,
            "source_domain": self.source_domain,
            "mode": self.mode,
            "assistance_level": self.assistance_level,
            "transformations": self.transformations,
            "verifier_agreement": self.verifier_agreement,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferenceProposal":
        if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported inference proposal schema version")
        frame = value.get("frame")
        if not isinstance(frame, Mapping):
            raise ValueError("inference frame must be an object")
        transformations = value.get("transformations")
        if transformations is not None and not isinstance(transformations, Mapping):
            raise ValueError("transformations must be an object or null")
        proposal = cls(
            proposal_id=_require_text(value["proposal_id"], "proposal_id"),
            source_id=_require_text(value["source_id"], "source_id"),
            minimal_span=_require_text(value["minimal_span"], "minimal_span"),
            surrounding_context=_require_text(
                value["surrounding_context"], "surrounding_context"
            ),
            query=_require_text(value["query"], "query"),
            main_proposition=_require_text(
                value["main_proposition"], "main_proposition"
            ),
            actor=_require_text(frame["actor"], "frame.actor"),
            predicate=_require_text(frame["predicate"], "frame.predicate"),
            patient=(str(frame["patient"]).strip() if frame.get("patient") else None),
            polarity=_require_text(frame["polarity"], "frame.polarity"),
            modality=_require_text(frame["modality"], "frame.modality"),
            scope_cues=tuple(map(str, frame.get("scope_cues", []))),
            axis_candidates=tuple(map(str, value.get("axis_candidates", []))),
            primary_axis=_require_text(value["primary_axis"], "primary_axis"),
            confidence=float(value["confidence"]),
            ambiguity_flags=tuple(map(str, value.get("ambiguity_flags", []))),
            rejection_reason=(
                str(value["rejection_reason"]).strip()
                if value.get("rejection_reason")
                else None
            ),
            predicate_family=_require_text(
                value["predicate_family"], "predicate_family"
            ),
            syntactic_form=_require_text(value["syntactic_form"], "syntactic_form"),
            source_domain=_require_text(value["source_domain"], "source_domain"),
            mode=_require_text(value["mode"], "mode"),
            assistance_level=_require_text(
                value["assistance_level"], "assistance_level"
            ),
            transformations=transformations,
            verifier_agreement=(
                bool(value["verifier_agreement"])
                if value.get("verifier_agreement") is not None
                else None
            ),
        )
        proposal.validate()
        return proposal


class InferenceAdapter(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def infer(self, source: SourceRecord) -> Sequence[InferenceProposal]: ...


class JsonlInferenceAdapter:
    def __init__(self, path: Path, *, producer_identity: str) -> None:
        proposals = [InferenceProposal.from_dict(row) for row in _read_jsonl(path)]
        self._by_source: dict[str, list[InferenceProposal]] = defaultdict(list)
        for proposal in proposals:
            self._by_source[proposal.source_id].append(proposal)
        self._path = path.resolve()
        self._producer_identity = producer_identity

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": "precomputed-jsonl",
            "producer_identity": self._producer_identity,
            "sampling_configuration": "producer-managed; unavailable to replay",
            "deterministic_replay": True,
            "proposal_sha256": _sha256_file(self._path),
        }

    def infer(self, source: SourceRecord) -> Sequence[InferenceProposal]:
        return tuple(self._by_source.get(source.source_id, ()))


class HeuristicInferenceAdapter:
    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": "deterministic-heuristic",
            "revision": "natural-question-v1",
            "temperature": 0,
        }

    def infer(self, source: SourceRecord) -> Sequence[InferenceProposal]:
        sentences = _sentences(source.text)
        if len(sentences) < 2 or not source.title:
            return ()
        cue_text = f"{source.title} {sentences[0]}".lower()
        if re.search(r"\b(?:may|might|must|should|could|allowed|required)\b", cue_text):
            axis = "modality"
        elif re.search(r"\b(?:all|every|none|only|some|any|except)\b", cue_text):
            axis = "scope"
        elif re.search(r"\b(?:by|from|to|against|between)\b", cue_text):
            axis = "direction"
        else:
            axis = "relation"
        words = _WORD.findall(sentences[0])
        if len(words) < 4:
            return ()
        span = sentences[0]
        context = " ".join(sentences[: min(4, len(sentences))])
        proposal = InferenceProposal(
            proposal_id=f"heuristic-{_sha256_text(source.source_id)[:20]}",
            source_id=source.source_id,
            minimal_span=span,
            surrounding_context=context,
            query=source.title if source.title.endswith("?") else source.title + "?",
            main_proposition=span,
            actor=words[0],
            predicate=words[1],
            patient=" ".join(words[2:4]),
            polarity="negative" if re.search(r"\b(?:not|never|no)\b", span, re.I) else "positive",
            modality="marked" if axis == "modality" else "asserted",
            scope_cues=tuple(
                sorted(set(re.findall(r"\b(?:all|every|none|only|some|any)\b", span, re.I)))
            ),
            axis_candidates=(axis,),
            primary_axis=axis,
            confidence=0.55,
            ambiguity_flags=("heuristic-frame",),
            rejection_reason=None,
            predicate_family=_slug(words[1]),
            syntactic_form="unknown",
            source_domain=_slug(source.site),
            mode="seed-only",
            assistance_level="query-frame-axis",
        )
        proposal.validate()
        return (proposal,)


@dataclass(frozen=True)
class MinedCandidate:
    candidate_id: str
    source: SourceRecord
    proposal: InferenceProposal

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "source": self.source.to_dict(),
            "proposal": self.proposal.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MinedCandidate":
        if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported mined candidate schema version")
        source = SourceRecord.from_dict(value["source"])
        proposal = InferenceProposal.from_dict(value["proposal"])
        if proposal.source_id != source.source_id:
            raise ValueError("candidate source and inference source do not match")
        return cls(
            candidate_id=_require_text(value["candidate_id"], "candidate_id"),
            source=source,
            proposal=proposal,
        )


def _sentences(value: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", value)
        if item.strip()
    ]


def _words(value: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(value) if len(word) > 2}


def _jaccard(left: str, right: str) -> float:
    left_words = _words(left)
    right_words = _words(right)
    if not left_words and not right_words:
        return 1.0
    return len(left_words & right_words) / max(1, len(left_words | right_words))


def source_rejection_reason(source: SourceRecord) -> str | None:
    try:
        source.validate()
    except ValueError:
        return "invalid-provenance"
    if source.content_license.strip().lower() == "unknown":
        return "missing-license"
    length = len(source.text)
    if length < 100:
        return "source-too-short"
    if length > 4000:
        return "source-too-long"
    if source.redactions:
        return "pii-or-url-redacted"
    if len(_sentences(source.text)) < 2:
        return "insufficient-context"
    if len(_WORD.findall(source.text)) < 25:
        return "fragment-or-nonprose"
    if re.search(r"(?:^|\n)(?:thanks|regards|cheers)[,!]?\s+[A-Z][a-z]+$", source.text):
        return "signature-noise"
    return None


def candidate_rejection_reason(
    source: SourceRecord,
    proposal: InferenceProposal,
    *,
    mode: str,
) -> str | None:
    if proposal.mode != mode:
        return "wrong-operating-mode"
    if proposal.rejection_reason:
        return f"inference:{proposal.rejection_reason}"
    if proposal.source_id != source.source_id:
        return "source-mismatch"
    if proposal.confidence < 0.60:
        return "low-confidence"
    if proposal.primary_axis not in SUPPORTED_AXES:
        return "unsupported-axis"
    if not proposal.query.endswith("?") or not 10 <= len(proposal.query) <= 240:
        return "unclear-query"
    if not 20 <= len(proposal.minimal_span) <= 900:
        return "invalid-span-length"
    if proposal.minimal_span not in source.text:
        return "span-not-in-source"
    if proposal.surrounding_context not in source.text:
        return "context-not-in-source"
    if proposal.minimal_span not in proposal.surrounding_context:
        return "span-not-in-context"
    if len(proposal.surrounding_context) <= len(proposal.minimal_span) + 25:
        return "no-irrelevant-control-context"
    if len(_sentences(proposal.surrounding_context)) < 2:
        return "no-irrelevant-control-context"
    if _LEAKAGE.search(
        " ".join(
            (proposal.query, proposal.minimal_span, proposal.surrounding_context)
        )
    ):
        return "evaluation-leakage"
    if _PLACEHOLDER_PREFIX in json.dumps(proposal.to_dict()).lower():
        return "placeholder"
    if any(
        flag in {"hidden-context", "private-information", "pure-code", "multi-axis"}
        for flag in proposal.ambiguity_flags
    ):
        return "blocking-ambiguity"
    if mode == "auto-proposal":
        assert proposal.transformations is not None
        transformed = []
        for role in ROLES:
            item = proposal.transformations[role]
            if not isinstance(item, Mapping):
                return "invalid-auto-transformation"
            text = str(item.get("transformed_text", "")).strip()
            if not text or text == proposal.surrounding_context:
                return "invalid-auto-transformation"
            if _PLACEHOLDER_PREFIX in text.lower():
                return "placeholder"
            transformed.append(text)
        if len(set(transformed)) != len(ROLES):
            return "duplicate-auto-transformation"
    return None


def natural_pair_hints(source: SourceRecord) -> list[dict[str, str]]:
    cues = re.compile(
        r"\b(?:actually|correction|however|rather than|not .+ but|to clarify|instead)\b",
        re.I,
    )
    return [
        {"source_id": source.source_id, "sentence": sentence}
        for sentence in _sentences(source.text)
        if cues.search(sentence)
    ]


def _length_bucket(value: str) -> str:
    length = len(_WORD.findall(value))
    if length < 60:
        return "short"
    if length < 140:
        return "medium"
    return "long"


def _select_diverse(
    candidates: Sequence[MinedCandidate],
    *,
    maximum: int,
    axis_target: int,
    seed: int,
) -> list[MinedCandidate]:
    if maximum < 1 or axis_target < 1:
        raise ValueError("selection targets must be positive")
    expected = axis_target * len(SUPPORTED_AXES)
    if maximum != expected:
        raise ValueError(
            f"maximum must equal the balanced axis target ({expected})"
        )
    rng = random.Random(seed)
    by_axis: dict[str, list[MinedCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_axis[candidate.proposal.primary_axis].append(candidate)
    missing = {
        axis: axis_target - len(by_axis.get(axis, ()))
        for axis in SUPPORTED_AXES
        if len(by_axis.get(axis, ())) < axis_target
    }
    if missing:
        raise ValueError(f"insufficient valid candidates for axis targets: {missing}")

    selected: list[MinedCandidate] = []
    site_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    predicates: set[str] = set()
    forms: set[str] = set()
    buckets: set[str] = set()
    for axis in ("relation", "direction", "scope", "modality"):
        pool = list(by_axis[axis])
        rng.shuffle(pool)
        for _ in range(axis_target):
            def score(candidate: MinedCandidate) -> tuple[float, str]:
                proposal = candidate.proposal
                novelty = 1.0
                if selected:
                    novelty = 1.0 - max(
                        _jaccard(
                            proposal.surrounding_context,
                            other.proposal.surrounding_context,
                        )
                        for other in selected
                    )
                value = (
                    3.0 / (1 + site_counts[candidate.source.site])
                    + 2.5 / (1 + domain_counts[proposal.source_domain])
                    + (1.5 if proposal.predicate_family not in predicates else 0.0)
                    + (1.0 if proposal.syntactic_form not in forms else 0.0)
                    + (
                        0.75
                        if _length_bucket(proposal.surrounding_context) not in buckets
                        else 0.0
                    )
                    + 3.0 * novelty
                    + 0.25 * proposal.confidence
                )
                return value, candidate.candidate_id

            chosen = max(pool, key=score)
            pool.remove(chosen)
            selected.append(chosen)
            site_counts[chosen.source.site] += 1
            domain_counts[chosen.proposal.source_domain] += 1
            predicates.add(chosen.proposal.predicate_family)
            forms.add(chosen.proposal.syntactic_form)
            buckets.add(_length_bucket(chosen.proposal.surrounding_context))
    return selected


def mine_sources(
    sources: Sequence[SourceRecord],
    inference: InferenceAdapter,
    *,
    mode: str,
    maximum: int,
    axis_target: int,
    seed: int,
) -> tuple[list[MinedCandidate], dict[str, Any]]:
    if mode not in SUPPORTED_MODES:
        raise ValueError("unsupported mining mode")
    source_rejections: Counter[str] = Counter()
    candidate_rejections: Counter[str] = Counter()
    usable = []
    for source in sources:
        reason = source_rejection_reason(source)
        if reason:
            source_rejections[reason] += 1
        else:
            usable.append(source)

    inferred: list[tuple[SourceRecord, InferenceProposal]] = []
    inferred_source_ids = set()
    for source in usable:
        source_proposals = tuple(inference.infer(source))
        if source_proposals:
            inferred_source_ids.add(source.source_id)
        for proposal in source_proposals:
            inferred.append((source, proposal))

    valid = []
    ambiguity_count = 0
    for source, proposal in inferred:
        if proposal.ambiguity_flags:
            ambiguity_count += 1
        reason = candidate_rejection_reason(source, proposal, mode=mode)
        if reason:
            candidate_rejections[reason] += 1
            continue
        candidate_id = f"candidate-{_sha256_text(proposal.proposal_id)[:24]}"
        valid.append(MinedCandidate(candidate_id, source, proposal))

    deduplicated = []
    duplicate_count = 0
    for candidate in sorted(
        valid,
        key=lambda item: (-item.proposal.confidence, item.candidate_id),
    ):
        if any(
            candidate.source.source_hash == other.source.source_hash
            or _jaccard(
                candidate.proposal.surrounding_context,
                other.proposal.surrounding_context,
            )
            >= 0.82
            for other in deduplicated
        ):
            duplicate_count += 1
            continue
        deduplicated.append(candidate)

    selected = _select_diverse(
        deduplicated,
        maximum=maximum,
        axis_target=axis_target,
        seed=seed,
    )
    raw_by_site = Counter(source.site for source in sources)
    usable_by_site = Counter(source.site for source in usable)
    inferred_by_site = Counter(
        source.site for source in usable if source.source_id in inferred_source_ids
    )
    selected_by_site = Counter(candidate.source.site for candidate in selected)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "mode": mode,
        "seed": seed,
        "inference_identity": dict(inference.identity),
        "funnel": {
            "raw_source_items": len(sources),
            "usable_text_items": len(usable),
            "inference_candidates": len(inferred),
            "deterministic_valid_candidates": len(valid),
            "deduplicated_candidates": len(deduplicated),
            "balanced_selected_candidates": len(selected),
        },
        "source_rejection_reasons": dict(sorted(source_rejections.items())),
        "candidate_rejection_reasons": dict(sorted(candidate_rejections.items())),
        "source_yield": {
            site: {
                "raw": raw_by_site[site],
                "usable": usable_by_site[site],
                "with_inference_candidate": inferred_by_site[site],
                "selected": selected_by_site[site],
            }
            for site in sorted(raw_by_site)
        },
        "selected_axis_distribution": dict(
            sorted(Counter(x.proposal.primary_axis for x in selected).items())
        ),
        "selected_site_distribution": dict(sorted(selected_by_site.items())),
        "selected_domain_distribution": dict(
            sorted(Counter(x.proposal.source_domain for x in selected).items())
        ),
        "duplicate_rate": duplicate_count / max(1, len(valid)),
        "inference_source_coverage_rate": len(inferred_source_ids)
        / max(1, len(usable)),
        "sources_without_inference_candidate": len(usable) - len(inferred_source_ids),
        "inference_ambiguity_rate": ambiguity_count / max(1, len(inferred)),
        "natural_pair_hint_count": sum(len(natural_pair_hints(x)) for x in sources),
    }
    if mode == "auto-proposal":
        report["selected_verifier_agreement_rate"] = sum(
            candidate.proposal.verifier_agreement is True for candidate in selected
        ) / max(1, len(selected))
    return selected, report


def _author_item(seed_id: str, axis: str, role: str) -> dict[str, Any]:
    relevant = role != "control"
    changed = role != "invariant"
    before = f"<human-{axis}-{role}-before>"
    return {
        "annotation_id": f"{seed_id}-{role}",
        "axis": axis,
        "role": role,
        "transformed_text": f"<human-{axis}-{role}-document>",
        "frame_id": "target" if relevant else "distractor",
        "query_relevant": relevant,
        "value_changed": changed,
        "before_value": before,
        "after_value": before if not changed else f"<human-{axis}-{role}-after>",
    }


def pack_candidates(
    candidates: Sequence[MinedCandidate],
    output_dir: Path,
    *,
    mode: str,
    protocol_version: str,
    author_slots: int,
    axis_target: int,
    seed: int,
    inference_identity: Mapping[str, Any],
    overwrite: bool,
) -> tuple[Path, ...]:
    if mode not in SUPPORTED_MODES:
        raise ValueError("unsupported packing mode")
    if not protocol_version.strip():
        raise ValueError("protocol_version cannot be empty")
    expected = axis_target * len(SUPPORTED_AXES)
    if len(candidates) != expected:
        raise ValueError(f"pack requires exactly {expected} balanced candidates")
    axis_counts = Counter(x.proposal.primary_axis for x in candidates)
    if set(axis_counts) != SUPPORTED_AXES or any(
        axis_counts[axis] != axis_target for axis in SUPPORTED_AXES
    ):
        raise ValueError("candidate axes are not exactly balanced")
    if len({x.source.site for x in candidates}) < 3:
        raise ValueError("source pack requires at least three source sites")
    if len({x.proposal.source_domain for x in candidates}) < 3:
        raise ValueError("source pack requires at least three source domains")
    if mode == "seed-only":
        domain_axes: dict[str, set[str]] = defaultdict(set)
        for candidate in candidates:
            domain_axes[candidate.proposal.source_domain].add(
                candidate.proposal.primary_axis
            )
        incomplete_domains = {
            domain: sorted(SUPPORTED_AXES - axes)
            for domain, axes in domain_axes.items()
            if axes != SUPPORTED_AXES
        }
        if incomplete_domains:
            raise ValueError(
                "each seed-only source domain must cover every axis: "
                f"{incomplete_domains}"
            )

    output_dir = output_dir.resolve()
    provenance = output_dir / "private-source-provenance.jsonl"
    manifest = output_dir / "source-seed-manifest.json"
    readme = output_dir / "README.md"
    if mode == "auto-proposal":
        packet_paths = (output_dir / "development-auto-proposals.jsonl",)
    else:
        if author_slots < 1:
            raise ValueError("author_slots must be positive")
        packet_paths = tuple(
            output_dir / f"author-slot-{slot:02d}.source-seeds.jsonl"
            for slot in range(1, author_slots + 1)
        )
    targets = (*packet_paths, provenance, manifest, readme)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    ordered = []
    for axis in ("relation", "direction", "scope", "modality"):
        axis_candidates = [x for x in candidates if x.proposal.primary_axis == axis]
        rng.shuffle(axis_candidates)
        ordered.extend(axis_candidates)

    provenance_rows = []
    if mode == "seed-only":
        packets: list[list[dict[str, Any]]] = [[] for _ in packet_paths]
        for index, candidate in enumerate(ordered):
            proposal = candidate.proposal
            seed_id = f"seed-{_sha256_text(candidate.candidate_id)[:20]}"
            slot = index % author_slots
            row = {
                "schema_version": 1,
                "example_only": True,
                "draft_status": "incomplete",
                "case_id": seed_id,
                "source_seed_id": seed_id,
                "inference_assisted_seed": True,
                "author_id_hash": "sha256:replace-with-64-lowercase-hex-digits",
                "source_group": _slug(proposal.source_domain, prefix="source-"),
                "collection_protocol": protocol_version,
                "language": "en",
                "context_text": proposal.query,
                "base_text": proposal.surrounding_context,
                "template_id": _slug(proposal.syntactic_form, prefix="natural-"),
                "predicate_family": _slug(proposal.predicate_family),
                "target_axis": proposal.primary_axis,
                "axis_authoring_contract": _AXIS_CONTRACTS[proposal.primary_axis],
                "items": [
                    _author_item(seed_id, proposal.primary_axis, role)
                    for role in ROLES
                ],
            }
            packets[slot].append(row)
            provenance_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "source_seed_id": seed_id,
                    "candidate_id": candidate.candidate_id,
                    "public_packet": packet_paths[slot].name,
                    "source": candidate.source.to_dict(),
                    "inference_proposal": proposal.to_dict(),
                }
            )
        for path, rows in zip(packet_paths, packets):
            _write_jsonl(path, rows)
    else:
        rows = []
        for candidate in ordered:
            proposal = candidate.proposal
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_id": candidate.candidate_id,
                    "mode": "auto-proposal",
                    "machine_generated": True,
                    "development_only": True,
                    "claim_eligible": False,
                    "model_evaluation_forbidden": True,
                    "source_seed_id": f"auto-{_sha256_text(candidate.candidate_id)[:20]}",
                    "query": proposal.query,
                    "source_text": proposal.surrounding_context,
                    "target_axis": proposal.primary_axis,
                    "transformations": proposal.transformations,
                    "verifier_agreement": proposal.verifier_agreement,
                }
            )
            provenance_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_id": candidate.candidate_id,
                    "source": candidate.source.to_dict(),
                    "inference_proposal": proposal.to_dict(),
                }
            )
        _write_jsonl(packet_paths[0], rows)
    _write_jsonl(provenance, provenance_rows)

    readme_text = f"""# Source-grounded frame seed kit

Mode: `{mode}`. Protocol: `{protocol_version}`.

The public author packets intentionally omit source URL, attribution, inference
confidence, frame proposal, and provenance. The private provenance file must
remain with the coordinator and must never be sent to blind reviewers.

In `seed-only` mode, inference supplied the natural source, query, and proposed
axis only. A human must independently write critical, control, and invariant
transformations. Do not use an LLM or expose model output to the author.

Do not run Spectra, MiniLM, the frame observer, or NLI on this material. This
dry run is discarded after authoring and blind-review workflow validation.
"""
    readme.write_text(readme_text, encoding="utf-8")
    _write_json(
        manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "mode": mode,
            "protocol_version": protocol_version,
            "source_grounded": True,
            "inference_assisted": True,
            "human_transformations_required": mode == "seed-only",
            "machine_generated": mode == "auto-proposal",
            "development_only": (
                mode == "auto-proposal" or "dry-run" in protocol_version
            ),
            "dry_run_only": "dry-run" in protocol_version,
            "claim_eligible": False,
            "model_evaluation_forbidden": True,
            "seed": seed,
            "case_count": len(candidates),
            "axis_counts": dict(sorted(axis_counts.items())),
            "site_counts": dict(
                sorted(Counter(x.source.site for x in candidates).items())
            ),
            "domain_counts": dict(
                sorted(Counter(x.proposal.source_domain for x in candidates).items())
            ),
            "inference_identity": dict(inference_identity),
            "verifier_agreement_rate": (
                sum(x.proposal.verifier_agreement is True for x in candidates)
                / max(1, len(candidates))
                if mode == "auto-proposal"
                else None
            ),
            "packet_sha256": {path.name: _sha256_file(path) for path in packet_paths},
            "private_provenance_sha256": _sha256_file(provenance),
        },
    )
    return (*packet_paths, provenance, manifest, readme)


def fetch_run(
    *,
    sites: Sequence[str],
    from_date: str | None,
    to_date: str | None,
    max_items: int,
    output: Path,
    manifest: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    if not sites:
        raise ValueError("at least one Stack Exchange site is required")
    if max_items < 1:
        raise ValueError("max_items must be positive")
    for path in (output, manifest):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite: {path}")
    adapter = StackExchangeAdapter()
    per_site = max(1, min(100, (max_items + len(sites) - 1) // len(sites)))
    records = []
    requests = []
    for site in sites:
        fetched, details = adapter.fetch(
            site=site,
            from_date=_parse_date(from_date),
            to_date=_parse_date(to_date, end=True),
            max_items=per_site,
        )
        records.extend(fetched)
        requests.append({"site": site, **details})
    records = records[:max_items]
    _write_jsonl(output, [record.to_dict() for record in records])
    _write_json(
        manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "adapter_identity": adapter.identity,
            "sites": list(sites),
            "from_date": from_date,
            "to_date": to_date,
            "max_items": max_items,
            "record_count": len(records),
            "requests": requests,
            "sha256": _sha256_file(output),
        },
    )
    return output.resolve(), manifest.resolve()


def mine_run(
    *,
    sources_path: Path,
    inference_provider: str,
    inference_proposals: Path | None,
    inference_identity: str,
    mode: str,
    maximum: int,
    axis_target: int,
    seed: int,
    output: Path,
    manifest: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    for path in (output, manifest):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite: {path}")
    sources = [SourceRecord.from_dict(row) for row in _read_jsonl(sources_path)]
    if inference_provider == "jsonl":
        if inference_proposals is None:
            raise ValueError("jsonl inference requires --inference-proposals")
        inference: InferenceAdapter = JsonlInferenceAdapter(
            inference_proposals,
            producer_identity=inference_identity,
        )
    elif inference_provider == "heuristic":
        inference = HeuristicInferenceAdapter()
    else:
        raise ValueError("unsupported inference provider")
    candidates, report = mine_sources(
        sources,
        inference,
        mode=mode,
        maximum=maximum,
        axis_target=axis_target,
        seed=seed,
    )
    _write_jsonl(output, [candidate.to_dict() for candidate in candidates])
    report.update(
        {
            "sources_sha256": _sha256_file(sources_path),
            "selected_sha256": _sha256_file(output),
            "selected_output": str(output.resolve()),
        }
    )
    _write_json(manifest, report)
    return output.resolve(), manifest.resolve()


def status_run(candidates_path: Path) -> dict[str, Any]:
    candidates = [
        MinedCandidate.from_dict(row) for row in _read_jsonl(candidates_path)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "candidate_count": len(candidates),
        "axis_counts": dict(
            sorted(Counter(x.proposal.primary_axis for x in candidates).items())
        ),
        "site_counts": dict(
            sorted(Counter(x.source.site for x in candidates).items())
        ),
        "domain_counts": dict(
            sorted(Counter(x.proposal.source_domain for x in candidates).items())
        ),
        "mode_counts": dict(
            sorted(Counter(x.proposal.mode for x in candidates).items())
        ),
        "ambiguity_rate": sum(bool(x.proposal.ambiguity_flags) for x in candidates)
        / max(1, len(candidates)),
        "sha256": _sha256_file(candidates_path),
    }


def pack_run(
    *,
    candidates_path: Path,
    mining_manifest: Path,
    output_dir: Path,
    mode: str,
    protocol_version: str,
    author_slots: int,
    axis_target: int,
    seed: int,
    overwrite: bool,
) -> tuple[Path, ...]:
    candidates = [
        MinedCandidate.from_dict(row) for row in _read_jsonl(candidates_path)
    ]
    mining = json.loads(mining_manifest.read_text(encoding="utf-8"))
    return pack_candidates(
        candidates,
        output_dir,
        mode=mode,
        protocol_version=protocol_version,
        author_slots=author_slots,
        axis_target=axis_target,
        seed=seed,
        inference_identity=mining["inference_identity"],
        overwrite=overwrite,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine natural public text into inference-assisted frame seeds."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch")
    fetch.add_argument("--adapter", choices=("stackexchange",), default="stackexchange")
    fetch.add_argument("--site", action="append", required=True)
    fetch.add_argument("--from-date")
    fetch.add_argument("--to-date")
    fetch.add_argument("--max-items", type=int, default=60)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--manifest", type=Path, required=True)
    fetch.add_argument("--overwrite", action="store_true")

    mine = commands.add_parser("mine")
    mine.add_argument("--sources", type=Path, required=True)
    mine.add_argument(
        "--inference-provider", choices=("jsonl", "heuristic"), required=True
    )
    mine.add_argument("--inference-proposals", type=Path)
    mine.add_argument("--inference-identity", default="unspecified")
    mine.add_argument("--mode", choices=tuple(sorted(SUPPORTED_MODES)), required=True)
    mine.add_argument("--max-retained", type=int, default=12)
    mine.add_argument("--axis-target", type=int, default=3)
    mine.add_argument("--seed", type=int, default=173205)
    mine.add_argument("--output", type=Path, required=True)
    mine.add_argument("--manifest", type=Path, required=True)
    mine.add_argument("--overwrite", action="store_true")

    status = commands.add_parser("status")
    status.add_argument("--candidates", type=Path, required=True)
    status.add_argument("--output", type=Path)
    status.add_argument("--overwrite", action="store_true")

    pack = commands.add_parser("pack")
    pack.add_argument("--candidates", type=Path, required=True)
    pack.add_argument("--mining-manifest", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    pack.add_argument("--mode", choices=tuple(sorted(SUPPORTED_MODES)), required=True)
    pack.add_argument("--protocol-version", required=True)
    pack.add_argument("--author-slots", type=int, default=3)
    pack.add_argument("--axis-target", type=int, default=3)
    pack.add_argument("--seed", type=int, default=223607)
    pack.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fetch":
        outputs = fetch_run(
            sites=args.site,
            from_date=args.from_date,
            to_date=args.to_date,
            max_items=args.max_items,
            output=args.output,
            manifest=args.manifest,
            overwrite=args.overwrite,
        )
    elif args.command == "mine":
        outputs = mine_run(
            sources_path=args.sources,
            inference_provider=args.inference_provider,
            inference_proposals=args.inference_proposals,
            inference_identity=args.inference_identity,
            mode=args.mode,
            maximum=args.max_retained,
            axis_target=args.axis_target,
            seed=args.seed,
            output=args.output,
            manifest=args.manifest,
            overwrite=args.overwrite,
        )
    elif args.command == "status":
        report = status_run(args.candidates)
        if args.output is None:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite: {args.output}")
        _write_json(args.output, report)
        outputs = (args.output.resolve(),)
    else:
        outputs = pack_run(
            candidates_path=args.candidates,
            mining_manifest=args.mining_manifest,
            output_dir=args.output_dir,
            mode=args.mode,
            protocol_version=args.protocol_version,
            author_slots=args.author_slots,
            axis_target=args.axis_target,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    for output in outputs:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
