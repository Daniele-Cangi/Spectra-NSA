from __future__ import annotations

import argparse
import hashlib
import json
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
            roles_by_axis.setdefault(change.axis.value, set()).add(role)
        if set(roles_by_axis) != REQUIRED_AXES:
            raise ValueError(f"{orbit.base_id}: incomplete human semantic axes")
        if any(roles != REQUIRED_ROLES for roles in roles_by_axis.values()):
            raise ValueError(f"{orbit.base_id}: incomplete matched human roles")


def run(
    input_path: Path,
    output: Path,
    *,
    protocol_version: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    if not protocol_version.strip():
        raise ValueError("protocol-version cannot be empty")
    output = output.resolve()
    manifest = output.with_name(f"{output.name}.manifest.json")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    orbits = _load_orbits(input_path)
    validate_human_orbits(orbits)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for orbit in orbits:
            handle.write(json.dumps(orbit.to_dict(), sort_keys=True) + "\n")
    temporary.replace(output)

    source_groups = Counter(str(x.metadata["source_group"]) for x in orbits)
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
                "orbit_count": len(orbits),
                "intervention_count": sum(len(x.interventions) for x in orbits),
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
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output, manifest = run(
        args.input,
        args.output,
        protocol_version=args.protocol_version,
        overwrite=args.overwrite,
    )
    print(f"wrote {output}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
