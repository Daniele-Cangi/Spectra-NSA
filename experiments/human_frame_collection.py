from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from spectra_v3.interventions import (
    Intervention,
    InterventionOrbit,
    VerificationStatus,
)
from spectra_v3.semantic_variables import SemanticAxis, SemanticVariableChange

from .human_frame_intake import HUMAN_SOURCE, REQUIRED_AXES, REQUIRED_ROLES


COLLECTION_SCHEMA_VERSION = 1
REQUIRED_REVIEWERS = 2
DRY_RUN_AUTHOR_SLOTS = 3
DRY_RUN_SOURCE_GROUPS = ("institutional", "dialogue", "narrative")
_AXIS_ORDER = ("relation", "direction", "scope", "modality")
_PLACEHOLDER_PREFIXES = ("<replace-", "<human-", "sha256:replace-")
_HASH_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOCAL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,95}\Z")


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a JSON boolean")
    return value


def _require_local_id(value: Any, field: str) -> str:
    resolved = str(value)
    if not _LOCAL_ID_PATTERN.fullmatch(resolved):
        raise ValueError(f"{field} is not a valid opaque local identifier")
    return resolved


def _require_hash_id(value: Any, field: str) -> str:
    resolved = str(value)
    if not _HASH_ID_PATTERN.fullmatch(resolved):
        raise ValueError(f"{field} must be a sha256 identity hash")
    return resolved


def _nonempty(value: Any, field: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{field} cannot be empty")
    return resolved


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class HumanDraftItem:
    annotation_id: str
    axis: SemanticAxis
    role: str
    transformed_text: str
    frame_id: str
    query_relevant: bool
    value_changed: bool
    before_value: str
    after_value: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HumanDraftItem":
        item = cls(
            annotation_id=_require_local_id(
                value["annotation_id"], "annotation_id"
            ),
            axis=SemanticAxis(str(value["axis"])),
            role=str(value["role"]),
            transformed_text=_nonempty(
                value["transformed_text"], "transformed_text"
            ),
            frame_id=_nonempty(value["frame_id"], "frame_id"),
            query_relevant=_require_bool(
                value["query_relevant"], "query_relevant"
            ),
            value_changed=_require_bool(value["value_changed"], "value_changed"),
            before_value=_nonempty(value["before_value"], "before_value"),
            after_value=_nonempty(value["after_value"], "after_value"),
        )
        item.validate_role_contract()
        return item

    @property
    def semantic_change(self) -> SemanticVariableChange:
        return SemanticVariableChange(
            axis=self.axis,
            frame_id=self.frame_id,
            query_relevant=self.query_relevant,
            value_changed=self.value_changed,
            before_value=self.before_value,
            after_value=self.after_value,
        )

    def validate_role_contract(self) -> None:
        if self.axis.value not in REQUIRED_AXES:
            raise ValueError(f"unsupported human axis: {self.axis.value}")
        if self.role not in REQUIRED_ROLES:
            raise ValueError(f"unsupported human role: {self.role}")
        expected = {
            "critical": (True, True),
            "control": (False, True),
            "invariant": (True, False),
        }[self.role]
        actual = (self.query_relevant, self.value_changed)
        if actual != expected:
            raise ValueError(
                f"{self.annotation_id}: role {self.role} disagrees with "
                "query relevance/value-change contract"
            )
        self.semantic_change.validate_relation(
            self.semantic_change.expected_relation
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "axis": self.axis.value,
            "role": self.role,
            "transformed_text": self.transformed_text,
            "frame_id": self.frame_id,
            "query_relevant": self.query_relevant,
            "value_changed": self.value_changed,
            "before_value": self.before_value,
            "after_value": self.after_value,
        }


@dataclass(frozen=True)
class HumanDraft:
    case_id: str
    author_id_hash: str
    source_group: str
    collection_protocol: str
    language: str
    context_text: str
    base_text: str
    template_id: str
    predicate_family: str
    target_axis: SemanticAxis
    items: tuple[HumanDraftItem, ...]
    source_seed_id: str | None = None
    inference_assisted_seed: bool = False

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, protocol_version: str
    ) -> "HumanDraft":
        if int(value.get("schema_version", -1)) != COLLECTION_SCHEMA_VERSION:
            raise ValueError("unsupported human draft schema version")
        if value.get("example_only") is True:
            raise ValueError("example-only draft cannot enter collection")
        if "machine_generated" in value and value["machine_generated"] is not False:
            raise ValueError("machine-generated draft cannot enter human collection")
        if "development_only" in value and value["development_only"] is not False:
            raise ValueError("development-only draft cannot enter human collection")
        if value.get("draft_status") != "complete":
            raise ValueError("human draft must be explicitly marked complete")
        draft = cls(
            case_id=_require_local_id(value["case_id"], "case_id"),
            author_id_hash=_require_hash_id(
                value["author_id_hash"], "author_id_hash"
            ),
            source_group=_require_local_id(value["source_group"], "source_group"),
            collection_protocol=_nonempty(
                value["collection_protocol"], "collection_protocol"
            ),
            language=_nonempty(value.get("language", "en"), "language"),
            context_text=_nonempty(value["context_text"], "context_text"),
            base_text=_nonempty(value["base_text"], "base_text"),
            template_id=_require_local_id(value["template_id"], "template_id"),
            predicate_family=_require_local_id(
                value["predicate_family"], "predicate_family"
            ),
            target_axis=SemanticAxis(str(value["target_axis"])),
            items=tuple(
                HumanDraftItem.from_dict(item) for item in value["items"]
            ),
            source_seed_id=(
                _require_local_id(value["source_seed_id"], "source_seed_id")
                if value.get("source_seed_id") is not None
                else None
            ),
            inference_assisted_seed=_require_bool(
                value.get("inference_assisted_seed", False),
                "inference_assisted_seed",
            ),
        )
        if draft.collection_protocol != protocol_version:
            raise ValueError(
                f"{draft.case_id}: collection protocol does not match frozen version"
            )
        draft.validate()
        return draft

    def validate(self) -> None:
        if self.inference_assisted_seed and self.source_seed_id is None:
            raise ValueError(
                f"{self.case_id}: inference-assisted draft requires source_seed_id"
            )
        if not self.items:
            raise ValueError(f"{self.case_id}: draft has no interventions")
        annotation_ids = [item.annotation_id for item in self.items]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError(f"{self.case_id}: duplicate annotation_id")
        transformed = [item.transformed_text for item in self.items]
        if self.base_text in transformed or len(transformed) != len(set(transformed)):
            raise ValueError(
                f"{self.case_id}: transformed texts must be unique and non-base"
            )
        if self.target_axis.value not in REQUIRED_AXES:
            raise ValueError(f"{self.case_id}: unsupported target axis")
        if {item.axis for item in self.items} != {self.target_axis}:
            raise ValueError(f"{self.case_id}: interventions cross semantic axes")
        roles = {item.role for item in self.items}
        if roles != REQUIRED_ROLES:
            raise ValueError(f"{self.case_id}: incomplete matched roles")
        if len(self.items) != len(REQUIRED_ROLES):
            raise ValueError(f"{self.case_id}: duplicate target-axis role")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "draft_status": "complete",
            "case_id": self.case_id,
            "author_id_hash": self.author_id_hash,
            "source_group": self.source_group,
            "collection_protocol": self.collection_protocol,
            "language": self.language,
            "context_text": self.context_text,
            "base_text": self.base_text,
            "template_id": self.template_id,
            "predicate_family": self.predicate_family,
            "target_axis": self.target_axis.value,
            "items": [item.to_dict() for item in self.items],
        }
        if self.source_seed_id is not None:
            result["source_seed_id"] = self.source_seed_id
            result["inference_assisted_seed"] = self.inference_assisted_seed
        return result


@dataclass(frozen=True)
class BlindReview:
    review_item_id: str
    reviewer_id_hash: str
    judged_axis: SemanticAxis
    judged_relation: str
    fluent: bool
    single_axis: bool
    accept: bool
    model_output_seen: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BlindReview":
        relation = str(value["judged_relation"])
        if relation not in {"change", "preserve"}:
            raise ValueError(f"unsupported judged relation: {relation}")
        return cls(
            review_item_id=_require_local_id(
                value["review_item_id"], "review_item_id"
            ),
            reviewer_id_hash=_require_hash_id(
                value["reviewer_id_hash"], "reviewer_id_hash"
            ),
            judged_axis=SemanticAxis(str(value["judged_axis"])),
            judged_relation=relation,
            fluent=_require_bool(value["fluent"], "fluent"),
            single_axis=_require_bool(value["single_axis"], "single_axis"),
            accept=_require_bool(value["accept"], "accept"),
            model_output_seen=_require_bool(
                value["model_output_seen"], "model_output_seen"
            ),
        )


def load_drafts(path: Path, *, protocol_version: str) -> list[HumanDraft]:
    drafts = [
        HumanDraft.from_dict(row, protocol_version=protocol_version)
        for row in _read_jsonl(path)
    ]
    _validate_unique_drafts(drafts)
    return drafts


def _validate_unique_drafts(drafts: Sequence[HumanDraft]) -> None:
    case_ids = [draft.case_id for draft in drafts]
    annotation_ids = [item.annotation_id for draft in drafts for item in draft.items]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be globally unique")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("annotation_id values must be globally unique")


def _placeholder_count(value: Any) -> int:
    if isinstance(value, str):
        return int(value.startswith(_PLACEHOLDER_PREFIXES))
    if isinstance(value, Mapping):
        return sum(_placeholder_count(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sum(_placeholder_count(item) for item in value)
    return 0


def inspect_draft_packets(
    draft_paths: Sequence[Path], *, protocol_version: str
) -> dict[str, Any]:
    """Report readiness and concrete gaps without accepting incomplete drafts."""

    if not draft_paths:
        raise ValueError("at least one draft packet is required")
    if not protocol_version.strip():
        raise ValueError("protocol-version cannot be empty")

    rows_with_paths: list[tuple[Path, dict[str, Any]]] = []
    validation_errors = []
    input_sha256 = {}
    for path in draft_paths:
        resolved = path.resolve()
        try:
            input_sha256[str(resolved)] = _sha256(resolved)
            rows = _read_jsonl(resolved)
        except (OSError, ValueError) as exc:
            validation_errors.append(
                {"path": str(resolved), "case_id": None, "reason": str(exc)}
            )
            continue
        rows_with_paths.extend((resolved, row) for row in rows)

    case_ids = Counter(str(row.get("case_id", "<missing>")) for _, row in rows_with_paths)
    annotation_ids = Counter(
        str(item.get("annotation_id", "<missing>"))
        for _, row in rows_with_paths
        for item in (
            row.get("items", [])
            if isinstance(row.get("items", []), Sequence)
            and not isinstance(row.get("items", []), (str, bytes))
            else []
        )
        if isinstance(item, Mapping)
    )
    status_counts = Counter(
        str(row.get("draft_status", "<missing>")) for _, row in rows_with_paths
    )
    axis_counts = Counter(
        str(row.get("target_axis", "<missing>")) for _, row in rows_with_paths
    )
    author_counts = Counter(
        str(row.get("author_id_hash", "<missing>")) for _, row in rows_with_paths
    )
    source_group_counts = Counter(
        str(row.get("source_group", "<missing>")) for _, row in rows_with_paths
    )
    example_only_count = sum(
        row.get("example_only") is True for _, row in rows_with_paths
    )
    placeholder_count = sum(
        _placeholder_count(row) for _, row in rows_with_paths
    )
    valid_case_count = 0
    for path, row in rows_with_paths:
        try:
            HumanDraft.from_dict(row, protocol_version=protocol_version)
        except (KeyError, TypeError, ValueError) as exc:
            validation_errors.append(
                {
                    "path": str(path),
                    "case_id": str(row.get("case_id", "<missing>")),
                    "reason": str(exc),
                }
            )
        else:
            valid_case_count += 1

    duplicate_case_ids = sorted(
        value for value, count in case_ids.items() if count > 1
    )
    duplicate_annotation_ids = sorted(
        value for value, count in annotation_ids.items() if count > 1
    )
    case_count = len(rows_with_paths)
    completed_case_count = status_counts.get("complete", 0)
    gaps = []
    if case_count == 0:
        gaps.append("no draft cases found")
    if completed_case_count != case_count:
        gaps.append(f"{case_count - completed_case_count} cases are not complete")
    if example_only_count:
        gaps.append(f"{example_only_count} cases are still example-only")
    if placeholder_count:
        gaps.append(f"{placeholder_count} placeholder values remain")
    if validation_errors:
        gaps.append(f"{len(validation_errors)} cases or files fail validation")
    if duplicate_case_ids:
        gaps.append(f"{len(duplicate_case_ids)} duplicate case identifiers")
    if duplicate_annotation_ids:
        gaps.append(
            f"{len(duplicate_annotation_ids)} duplicate annotation identifiers"
        )

    return {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": protocol_version,
        "ready_for_assembly": not gaps,
        "case_count": case_count,
        "completed_case_count": completed_case_count,
        "valid_case_count": valid_case_count,
        "example_only_count": example_only_count,
        "placeholder_count": placeholder_count,
        "status_counts": dict(sorted(status_counts.items())),
        "axis_counts": dict(sorted(axis_counts.items())),
        "author_counts": dict(sorted(author_counts.items())),
        "source_group_counts": dict(sorted(source_group_counts.items())),
        "duplicate_case_ids": duplicate_case_ids,
        "duplicate_annotation_ids": duplicate_annotation_ids,
        "validation_errors": validation_errors,
        "gaps": gaps,
        "input_sha256": dict(sorted(input_sha256.items())),
    }


def assemble_draft_packets(
    draft_paths: Sequence[Path],
    output: Path,
    *,
    protocol_version: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Validate author packets jointly and emit one canonical pre-review file."""

    output = output.resolve()
    manifest = output.with_name(f"{output.name}.manifest.json")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    report = inspect_draft_packets(
        draft_paths, protocol_version=protocol_version
    )
    if not report["ready_for_assembly"]:
        raise ValueError("draft packets are not ready: " + "; ".join(report["gaps"]))

    drafts = [
        HumanDraft.from_dict(row, protocol_version=protocol_version)
        for path in draft_paths
        for row in _read_jsonl(path)
    ]
    _validate_unique_drafts(drafts)
    axis_counts = Counter(draft.target_axis.value for draft in drafts)
    author_counts = Counter(draft.author_id_hash for draft in drafts)
    source_group_counts = Counter(draft.source_group for draft in drafts)
    input_sha256 = {
        str(path.resolve()): _sha256(path.resolve()) for path in draft_paths
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, [draft.to_dict() for draft in drafts])
    manifest.write_text(
        json.dumps(
            {
                "schema_version": COLLECTION_SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_version": protocol_version,
                "human_authored": True,
                "evaluation_partition": "pre-review",
                "model_evaluation_forbidden": True,
                "locked": False,
                "claim_eligible": False,
                "case_count": len(drafts),
                "intervention_count": sum(len(draft.items) for draft in drafts),
                "axis_counts": dict(sorted(axis_counts.items())),
                "author_counts": dict(sorted(author_counts.items())),
                "source_group_counts": dict(sorted(source_group_counts.items())),
                "input_sha256": dict(sorted(input_sha256.items())),
                "sha256": _sha256(output),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output, manifest


def _blank_dry_run_case(
    *,
    author_slot: int,
    axis: str,
    source_group: str,
    protocol_version: str,
) -> dict[str, Any]:
    case_id = f"dry-a{author_slot:02d}-{axis}"
    items = []
    for role in ("critical", "control", "invariant"):
        relevant = role != "control"
        changed = role != "invariant"
        before = f"<replace-{axis}-{role}-before>"
        after = before if not changed else f"<replace-{axis}-{role}-after>"
        items.append(
            {
                "annotation_id": f"{case_id}-{role}",
                "axis": axis,
                "role": role,
                "transformed_text": f"<replace-{axis}-{role}-document>",
                "frame_id": "target" if relevant else "distractor",
                "query_relevant": relevant,
                "value_changed": changed,
                "before_value": before,
                "after_value": after,
            }
        )
    return {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "example_only": True,
        "draft_status": "incomplete",
        "case_id": case_id,
        "author_id_hash": "sha256:replace-with-64-lowercase-hex-digits",
        "source_group": source_group,
        "collection_protocol": protocol_version,
        "language": "en",
        "context_text": "<replace-with-human-written-query>",
        "base_text": "<replace-with-human-written-base-document>",
        "template_id": "replace-template",
        "predicate_family": "replace-predicate",
        "target_axis": axis,
        "items": items,
    }


def make_dry_run_kit(
    output_dir: Path,
    *,
    protocol_version: str,
    seed: int,
    overwrite: bool,
) -> tuple[Path, ...]:
    """Create twelve deliberately incomplete authoring cases without model data."""

    if not protocol_version.strip():
        raise ValueError("protocol-version cannot be empty")
    output_dir = output_dir.resolve()
    packet_paths = tuple(
        output_dir / f"author-slot-{slot:02d}.drafts.jsonl"
        for slot in range(1, DRY_RUN_AUTHOR_SLOTS + 1)
    )
    readme = output_dir / "README.md"
    manifest = output_dir / "dry-run-manifest.json"
    targets = (*packet_paths, readme, manifest)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    all_rows = []
    for slot, packet_path in enumerate(packet_paths, start=1):
        axes = list(_AXIS_ORDER)
        rng.shuffle(axes)
        rows = []
        for axis in axes:
            canonical_index = _AXIS_ORDER.index(axis)
            source_group = DRY_RUN_SOURCE_GROUPS[
                (slot - 1 + canonical_index) % len(DRY_RUN_SOURCE_GROUPS)
            ]
            rows.append(
                _blank_dry_run_case(
                    author_slot=slot,
                    axis=axis,
                    source_group=source_group,
                    protocol_version=protocol_version,
                )
            )
        _write_jsonl(packet_path, rows)
        all_rows.extend(rows)

    readme_text = """# Human frame dry run

This kit contains twelve incomplete, model-blind authoring cases: one case per
semantic axis for each of three author slots. It is a workflow test, not part of
the locked evaluation set.

For every assigned case:

1. write a natural query and base document;
2. write exactly three transformed documents for the assigned axis;
3. replace all placeholder values and the pseudonymous author hash;
4. remove `example_only` and set `draft_status` to `complete`;
5. do not use an LLM, machine paraphraser, synthetic-v4 text, or model output.

The coordinator validates the completed JSONL packets with `draft-status`,
assembles them with `assemble-drafts`, then creates the blind review packet.
Dry-run cases are discarded after the workflow and must never be frozen as
evaluation evidence.

Check the three returned packets without modifying them:

```
spectra-phase0-human-frame-collection draft-status --drafts \
  author-slot-01.drafts.jsonl author-slot-02.drafts.jsonl \
  author-slot-03.drafts.jsonl --protocol-version human-frame-v1-dry-run
```

Only a report with `ready_for_assembly: true` may be passed to
`assemble-drafts`.
"""
    readme_tmp = readme.with_name(f"{readme.name}.tmp")
    readme_tmp.write_text(readme_text, encoding="utf-8")
    readme_tmp.replace(readme)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": COLLECTION_SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "dry_run_only": True,
                "model_evaluation_forbidden": True,
                "protocol_version": protocol_version,
                "seed": seed,
                "author_slot_count": DRY_RUN_AUTHOR_SLOTS,
                "case_count": len(all_rows),
                "cases_per_axis": dict(
                    sorted(Counter(row["target_axis"] for row in all_rows).items())
                ),
                "cases_per_source_group": dict(
                    sorted(Counter(row["source_group"] for row in all_rows).items())
                ),
                "packet_sha256": {
                    path.name: _sha256(path) for path in packet_paths
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (*packet_paths, readme, manifest)


def make_blind_review_packet(
    drafts_path: Path,
    packet_path: Path,
    mapping_path: Path,
    *,
    protocol_version: str,
    seed: int,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    packet_path = packet_path.resolve()
    mapping_path = mapping_path.resolve()
    manifest = mapping_path.with_name(f"{mapping_path.name}.manifest.json")
    targets = (packet_path, mapping_path, manifest)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    if packet_path == mapping_path:
        raise ValueError("public packet and private mapping must be different files")
    drafts = load_drafts(drafts_path, protocol_version=protocol_version)
    packet_rows = []
    mapping_rows = []
    for draft in drafts:
        for item in draft.items:
            digest = hashlib.sha256(
                f"{protocol_version}:{seed}:{item.annotation_id}".encode("utf-8")
            ).hexdigest()[:32]
            review_item_id = f"review-{digest}"
            packet_rows.append(
                {
                    "schema_version": COLLECTION_SCHEMA_VERSION,
                    "review_item_id": review_item_id,
                    "language": draft.language,
                    "context_text": draft.context_text,
                    "base_text": draft.base_text,
                    "transformed_text": item.transformed_text,
                }
            )
            mapping_rows.append(
                {
                    "review_item_id": review_item_id,
                    "annotation_id": item.annotation_id,
                    "case_id": draft.case_id,
                }
            )
    random.Random(seed).shuffle(packet_rows)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(packet_path, packet_rows)
    _write_jsonl(mapping_path, mapping_rows)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": COLLECTION_SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_version": protocol_version,
                "seed": seed,
                "item_count": len(packet_rows),
                "blinded_fields": [
                    "annotation_id",
                    "author_id_hash",
                    "axis",
                    "role",
                    "expected_relation",
                    "model_output",
                ],
                "drafts_sha256": _sha256(drafts_path),
                "packet_sha256": _sha256(packet_path),
                "private_mapping_sha256": _sha256(mapping_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return packet_path, mapping_path, manifest


def _mapping_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    review_to_annotation = {}
    annotation_to_review = {}
    for row in rows:
        review_id = _require_local_id(row["review_item_id"], "review_item_id")
        annotation_id = _require_local_id(row["annotation_id"], "annotation_id")
        if review_id in review_to_annotation or annotation_id in annotation_to_review:
            raise ValueError("private review mapping is not one-to-one")
        review_to_annotation[review_id] = annotation_id
        annotation_to_review[annotation_id] = review_id
    return review_to_annotation, annotation_to_review


def compile_reviewed_orbits(
    drafts: Sequence[HumanDraft],
    mapping_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    *,
    review_protocol: str,
    mapping_sha256: str,
) -> list[InterventionOrbit]:
    if not review_protocol.strip():
        raise ValueError("review protocol cannot be empty")
    review_to_annotation, annotation_to_review = _mapping_lookup(mapping_rows)
    items = {
        item.annotation_id: (draft, item)
        for draft in drafts
        for item in draft.items
    }
    if set(annotation_to_review) != set(items):
        raise ValueError("private mapping does not exactly cover draft annotations")
    reviews = [BlindReview.from_dict(row) for row in review_rows]
    authors = {draft.author_id_hash for draft in drafts}
    grouped: dict[str, list[BlindReview]] = defaultdict(list)
    seen = set()
    for review in reviews:
        if review.review_item_id not in review_to_annotation:
            raise ValueError(f"unknown blind review item: {review.review_item_id}")
        key = (review.review_item_id, review.reviewer_id_hash)
        if key in seen:
            raise ValueError("duplicate review by the same reviewer")
        seen.add(key)
        if review.reviewer_id_hash in authors:
            raise ValueError("reviewer cannot author any case in the same locked batch")
        grouped[review.review_item_id].append(review)

    accepted_reviewers: dict[str, tuple[str, ...]] = {}
    for annotation_id, (draft, item) in items.items():
        review_id = annotation_to_review[annotation_id]
        item_reviews = grouped.get(review_id, [])
        reviewers = {review.reviewer_id_hash for review in item_reviews}
        if len(reviewers) < REQUIRED_REVIEWERS:
            raise ValueError(
                f"{annotation_id}: requires {REQUIRED_REVIEWERS} independent reviews"
            )
        expected_relation = item.semantic_change.expected_relation.value
        for review in item_reviews:
            if review.model_output_seen:
                raise ValueError(f"{annotation_id}: reviewer saw model output")
            if not review.accept or not review.fluent or not review.single_axis:
                raise ValueError(f"{annotation_id}: review rejected the item")
            if review.judged_axis != item.axis:
                raise ValueError(f"{annotation_id}: reviewers disagree on axis")
            if review.judged_relation != expected_relation:
                raise ValueError(
                    f"{annotation_id}: reviewers disagree on task relation"
                )
        accepted_reviewers[annotation_id] = tuple(sorted(reviewers))

    orbits = []
    for draft in drafts:
        interventions = tuple(
            Intervention(
                base_text=draft.base_text,
                transformed_text=item.transformed_text,
                family=item.axis.value,
                expected_relation=item.semantic_change.expected_relation,
                strength=1.0,
                generator_id=(
                    f"human.{draft.case_id}.{item.axis.value}.{item.role}"
                ),
                verification_status=VerificationStatus.HUMAN,
                metadata={
                    "adversarial_role": item.role,
                    "human_annotation_id": item.annotation_id,
                    "semantic_change": item.semantic_change.to_dict(),
                    "blinded_review": True,
                    "reviewer_count": len(accepted_reviewers[item.annotation_id]),
                    "reviewer_id_hashes": list(
                        accepted_reviewers[item.annotation_id]
                    ),
                    "review_protocol": review_protocol,
                    "private_mapping_sha256": mapping_sha256,
                },
            )
            for item in draft.items
        )
        fingerprint = hashlib.sha256(
            (
                f"{draft.case_id}:{draft.author_id_hash}:{draft.source_group}:"
                f"{draft.context_text}:{draft.base_text}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        orbits.append(
            InterventionOrbit(
                base_id=f"human-{draft.case_id}-{fingerprint}",
                base_text=draft.base_text,
                context_text=draft.context_text,
                interventions=interventions,
                source=HUMAN_SOURCE,
                metadata={
                    "human_authored": True,
                    "claim_eligible": False,
                    "development_only": False,
                    "evaluation_partition": "human-locked",
                    "author_id_hash": draft.author_id_hash,
                    "source_group": draft.source_group,
                    "collection_protocol": draft.collection_protocol,
                    "review_protocol": review_protocol,
                    "template_id": draft.template_id,
                    "predicate_family": draft.predicate_family,
                    "target_axis": draft.target_axis.value,
                    "source_seed_id": draft.source_seed_id,
                    "inference_assisted_seed": draft.inference_assisted_seed,
                    "language": draft.language,
                    "factor_fingerprint": fingerprint,
                },
            )
        )
    return orbits


def compile_run(
    drafts_path: Path,
    mapping_path: Path,
    review_paths: Sequence[Path],
    output: Path,
    *,
    collection_protocol: str,
    review_protocol: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    output = output.resolve()
    manifest = output.with_name(f"{output.name}.manifest.json")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    drafts = load_drafts(drafts_path, protocol_version=collection_protocol)
    mapping_rows = _read_jsonl(mapping_path)
    review_rows = [row for path in review_paths for row in _read_jsonl(path)]
    orbits = compile_reviewed_orbits(
        drafts,
        mapping_rows,
        review_rows,
        review_protocol=review_protocol,
        mapping_sha256=_sha256(mapping_path),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, [orbit.to_dict() for orbit in orbits])
    manifest.write_text(
        json.dumps(
            {
                "schema_version": COLLECTION_SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "human_authored": True,
                "locked": False,
                "claim_eligible": False,
                "collection_protocol": collection_protocol,
                "review_protocol": review_protocol,
                "required_reviewers": REQUIRED_REVIEWERS,
                "orbit_count": len(orbits),
                "intervention_count": sum(len(x.interventions) for x in orbits),
                "author_count": len({x.author_id_hash for x in drafts}),
                "source_group_counts": dict(
                    sorted(Counter(x.source_group for x in drafts).items())
                ),
                "drafts_sha256": _sha256(drafts_path),
                "private_mapping_sha256": _sha256(mapping_path),
                "review_sha256": {
                    str(path.resolve()): _sha256(path) for path in review_paths
                },
                "sha256": _sha256(output),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build blind-review packets and compile human frame orbits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_run = subparsers.add_parser("dry-run-kit")
    dry_run.add_argument("--output-dir", type=Path, required=True)
    dry_run.add_argument("--protocol-version", required=True)
    dry_run.add_argument("--seed", type=int, default=141421)
    dry_run.add_argument("--overwrite", action="store_true")

    status = subparsers.add_parser("draft-status")
    status.add_argument("--drafts", type=Path, nargs="+", required=True)
    status.add_argument("--protocol-version", required=True)
    status.add_argument("--output", type=Path)
    status.add_argument("--overwrite", action="store_true")

    assemble = subparsers.add_parser("assemble-drafts")
    assemble.add_argument("--drafts", type=Path, nargs="+", required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--protocol-version", required=True)
    assemble.add_argument("--overwrite", action="store_true")

    packet = subparsers.add_parser("review-packet")
    packet.add_argument("--drafts", type=Path, required=True)
    packet.add_argument("--packet", type=Path, required=True)
    packet.add_argument("--private-mapping", type=Path, required=True)
    packet.add_argument("--protocol-version", required=True)
    packet.add_argument("--seed", type=int, default=161803)
    packet.add_argument("--overwrite", action="store_true")

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--drafts", type=Path, required=True)
    compile_parser.add_argument("--private-mapping", type=Path, required=True)
    compile_parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--collection-protocol", required=True)
    compile_parser.add_argument("--review-protocol", required=True)
    compile_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dry-run-kit":
        outputs = make_dry_run_kit(
            args.output_dir,
            protocol_version=args.protocol_version,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    elif args.command == "draft-status":
        report = inspect_draft_packets(
            args.drafts,
            protocol_version=args.protocol_version,
        )
        if args.output is None:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        status_output = args.output.resolve()
        if status_output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite: {status_output}")
        _write_json(status_output, report)
        outputs = (status_output,)
    elif args.command == "assemble-drafts":
        outputs = assemble_draft_packets(
            args.drafts,
            args.output,
            protocol_version=args.protocol_version,
            overwrite=args.overwrite,
        )
    elif args.command == "review-packet":
        outputs = make_blind_review_packet(
            args.drafts,
            args.packet,
            args.private_mapping,
            protocol_version=args.protocol_version,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    else:
        outputs = compile_run(
            args.drafts,
            args.private_mapping,
            args.reviews,
            args.output,
            collection_protocol=args.collection_protocol,
            review_protocol=args.review_protocol,
            overwrite=args.overwrite,
        )
    for output in outputs:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
