from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from experiments.phase0_late_interaction import _load_orbits
from spectra_v3.interventions import InterventionOrbit, VerificationStatus
from spectra_v3.semantic_variables import semantic_change_from_metadata


REQUIRED_AXES = frozenset({"direction", "modality", "relation", "scope"})
REQUIRED_ROLES = frozenset({"control", "critical", "invariant"})
HUMAN_SOURCE = "human-authored-frame-pilot-v4"
MINIMUM_HUMAN_ORBITS = 96
MINIMUM_HUMAN_AUTHORS = 3
MINIMUM_SOURCE_GROUPS = 3
MINIMUM_REVIEWERS = 2
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_HASH_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


def validate_human_orbits(orbits: Sequence[InterventionOrbit]) -> None:
    """Validate provenance and matched annotations before freezing a human set."""

    if not orbits:
        raise ValueError("human intake cannot be empty")
    for orbit in orbits:
        if orbit.source != HUMAN_SOURCE:
            raise ValueError(f"{orbit.base_id}: source is not human-authored")
        metadata = orbit.metadata
        if metadata.get("human_authored") is not True:
            raise ValueError(f"{orbit.base_id}: human_authored must be true")
        if metadata.get("evaluation_partition") != "human-locked":
            raise ValueError(f"{orbit.base_id}: partition must be human-locked")
        for field in ("author_id_hash", "source_group", "collection_protocol"):
            if not str(metadata.get(field, "")).strip():
                raise ValueError(f"{orbit.base_id}: missing provenance field {field}")
        author_id_hash = str(metadata["author_id_hash"])
        if not _HASH_ID_PATTERN.fullmatch(author_id_hash):
            raise ValueError(f"{orbit.base_id}: author identity is not hashed")
        review_protocol = str(metadata.get("review_protocol", ""))
        if not review_protocol:
            raise ValueError(f"{orbit.base_id}: missing review protocol")

        roles_by_axis: dict[str, set[str]] = {}
        annotation_ids = set()
        for item in orbit.interventions:
            if item.verification_status != VerificationStatus.HUMAN:
                raise ValueError(
                    f"{orbit.base_id}: intervention is not human verified"
                )
            change = semantic_change_from_metadata(item.metadata)
            change.validate_relation(item.expected_relation)
            role = str(item.metadata.get("adversarial_role", ""))
            if role not in REQUIRED_ROLES:
                raise ValueError(f"{orbit.base_id}: invalid adversarial role {role}")
            annotation_id = str(item.metadata.get("human_annotation_id", ""))
            if not annotation_id or annotation_id in annotation_ids:
                raise ValueError(
                    f"{orbit.base_id}: human_annotation_id must be unique"
                )
            annotation_ids.add(annotation_id)
            if item.metadata.get("blinded_review") is not True:
                raise ValueError(f"{orbit.base_id}: review was not blinded")
            reviewers = item.metadata.get("reviewer_id_hashes")
            if not isinstance(reviewers, list):
                raise ValueError(f"{orbit.base_id}: reviewer hashes must be a list")
            if (
                len(set(map(str, reviewers))) < MINIMUM_REVIEWERS
                or int(item.metadata.get("reviewer_count", -1))
                != len(set(map(str, reviewers)))
            ):
                raise ValueError(f"{orbit.base_id}: insufficient independent reviews")
            if author_id_hash in reviewers:
                raise ValueError(f"{orbit.base_id}: author cannot review own item")
            if any(not _HASH_ID_PATTERN.fullmatch(str(value)) for value in reviewers):
                raise ValueError(f"{orbit.base_id}: reviewer identity is not hashed")
            if str(item.metadata.get("review_protocol", "")) != review_protocol:
                raise ValueError(f"{orbit.base_id}: inconsistent review protocol")
            mapping_hash = str(item.metadata.get("private_mapping_sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", mapping_hash):
                raise ValueError(f"{orbit.base_id}: missing private mapping hash")
            roles_by_axis.setdefault(change.axis.value, set()).add(role)
        if set(roles_by_axis) != REQUIRED_AXES:
            raise ValueError(f"{orbit.base_id}: incomplete human semantic axes")
        if any(roles != REQUIRED_ROLES for roles in roles_by_axis.values()):
            raise ValueError(f"{orbit.base_id}: incomplete matched human roles")
        if len(orbit.interventions) != len(REQUIRED_AXES) * len(REQUIRED_ROLES):
            raise ValueError(f"{orbit.base_id}: duplicate human axis-role item")


def run(
    input_path: Path,
    output: Path,
    *,
    protocol_version: str,
    evaluation_commit: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    if not protocol_version.strip():
        raise ValueError("protocol-version cannot be empty")
    if not _GIT_COMMIT_PATTERN.fullmatch(evaluation_commit):
        raise ValueError("evaluation-commit must be a full immutable git SHA")
    output = output.resolve()
    manifest = output.with_name(f"{output.name}.manifest.json")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    orbits = _load_orbits(input_path)
    validate_human_orbits(orbits)
    authors = {str(x.metadata["author_id_hash"]) for x in orbits}
    source_groups = Counter(str(x.metadata["source_group"]) for x in orbits)
    if len(orbits) < MINIMUM_HUMAN_ORBITS:
        raise ValueError(
            f"locked set requires at least {MINIMUM_HUMAN_ORBITS} human orbits"
        )
    if len(authors) < MINIMUM_HUMAN_AUTHORS:
        raise ValueError(
            f"locked set requires at least {MINIMUM_HUMAN_AUTHORS} authors"
        )
    if len(source_groups) < MINIMUM_SOURCE_GROUPS:
        raise ValueError(
            f"locked set requires at least {MINIMUM_SOURCE_GROUPS} source groups"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for orbit in orbits:
            handle.write(json.dumps(orbit.to_dict(), sort_keys=True) + "\n")
    temporary.replace(output)

    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "locked": True,
                "human_authored": True,
                "development_only": False,
                "claim_eligible": False,
                "claim_eligibility_note": (
                    "Eligibility requires a separately approved collection and "
                    "evaluation protocol; freezing alone is insufficient."
                ),
                "protocol_version": protocol_version,
                "evaluation_commit": evaluation_commit,
                "orbit_count": len(orbits),
                "intervention_count": sum(len(x.interventions) for x in orbits),
                "author_count": len(authors),
                "source_group_counts": dict(sorted(source_groups.items())),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
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
        description="Validate and freeze genuinely human-authored frame orbits."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output, manifest = run(
        args.input,
        args.output,
        protocol_version=args.protocol_version,
        evaluation_commit=args.evaluation_commit,
        overwrite=args.overwrite,
    )
    print(f"wrote {output}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
