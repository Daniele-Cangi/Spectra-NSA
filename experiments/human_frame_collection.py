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
    items: tuple[HumanDraftItem, ...]

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, protocol_version: str
    ) -> "HumanDraft":
        if int(value.get("schema_version", -1)) != COLLECTION_SCHEMA_VERSION:
            raise ValueError("unsupported human draft schema version")
        if value.get("example_only") is True:
            raise ValueError("example-only draft cannot enter collection")
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
            items=tuple(
                HumanDraftItem.from_dict(item) for item in value["items"]
            ),
        )
        if draft.collection_protocol != protocol_version:
            raise ValueError(
                f"{draft.case_id}: collection protocol does not match frozen version"
            )
        draft.validate()
        return draft

    def validate(self) -> None:
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
        roles_by_axis: dict[str, set[str]] = defaultdict(set)
        for item in self.items:
            roles_by_axis[item.axis.value].add(item.role)
        if set(roles_by_axis) != REQUIRED_AXES:
            raise ValueError(f"{self.case_id}: incomplete semantic axes")
        if any(roles != REQUIRED_ROLES for roles in roles_by_axis.values()):
            raise ValueError(f"{self.case_id}: incomplete matched roles")
        if len(self.items) != len(REQUIRED_AXES) * len(REQUIRED_ROLES):
            raise ValueError(f"{self.case_id}: duplicate axis-role item")


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
    case_ids = [draft.case_id for draft in drafts]
    annotation_ids = [item.annotation_id for draft in drafts for item in draft.items]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be globally unique")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("annotation_id values must be globally unique")
    return drafts


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
    if args.command == "review-packet":
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
