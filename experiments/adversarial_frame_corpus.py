from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from spectra_v3.interventions import (
    Intervention,
    InterventionOrbit,
    VerificationStatus,
    make_exact_replacement,
    validate_unique_generator_ids,
)
from spectra_v3.semantic_variables import SemanticAxis, SemanticVariableChange


ACTORS = (
    "Aster Labs",
    "Boreal Systems",
    "Cobalt Group",
    "Delta Works",
    "Ember Research",
    "Fjord Dynamics",
    "Granite Partners",
    "Helios Network",
    "Ionis Studio",
    "Jasper Holdings",
    "Kepler Ventures",
    "Lumen Institute",
)
PATIENTS = (
    "Project Juniper",
    "Atlas Program",
    "Orion Initiative",
    "Nimbus Platform",
    "Cedar Protocol",
    "Vega Project",
    "Solstice Program",
    "Meridian Initiative",
    "Aurora Platform",
    "Willow Protocol",
    "Pioneer Project",
    "Summit Program",
)


@dataclass(frozen=True)
class PredicateFamily:
    family_id: str
    lemma: str
    past: str
    paraphrase_lemma: str
    paraphrase_past: str
    contrast_lemma: str
    contrast_past: str


PREDICATE_FAMILIES = (
    PredicateFamily(
        "permission", "approve", "approved", "authorize", "authorized", "reject", "rejected"
    ),
    PredicateFamily(
        "support", "support", "supported", "endorse", "endorsed", "oppose", "opposed"
    ),
    PredicateFamily(
        "acquisition", "acquire", "acquired", "purchase", "purchased", "sell", "sold"
    ),
    PredicateFamily(
        "prevention", "prevent", "prevented", "stop", "stopped", "enable", "enabled"
    ),
    PredicateFamily(
        "funding", "fund", "funded", "finance", "financed", "defund", "defunded"
    ),
    PredicateFamily(
        "selection", "select", "selected", "choose", "chose", "exclude", "excluded"
    ),
    PredicateFamily(
        "launch", "launch", "launched", "introduce", "introduced", "cancel", "cancelled"
    ),
    PredicateFamily(
        "verification", "verify", "verified", "confirm", "confirmed", "dispute", "disputed"
    ),
)

TEMPLATE_IDS = (
    "active-leading",
    "active-reported",
    "passive-leading",
    "passive-records",
)
HELD_PREDICATE_FAMILIES = frozenset({"launch", "verification"})
HELD_TEMPLATE_IDS = frozenset({"passive-records"})


def evaluation_partition(predicate_family: str, template_id: str) -> str:
    predicate_held = predicate_family in HELD_PREDICATE_FAMILIES
    template_held = template_id in HELD_TEMPLATE_IDS
    if predicate_held and template_held:
        return "double-holdout"
    if predicate_held:
        return "held-predicate"
    if template_held:
        return "held-template"
    return "development"


def _is_passive(template_id: str) -> bool:
    return template_id.startswith("passive-")


def _decorate(template_id: str, clause: str) -> str:
    if template_id == "active-reported":
        return f"According to the audit, {clause}"
    if template_id == "passive-records":
        return f"The audit confirms that {clause}"
    return clause


def _clause(
    template_id: str,
    actor: str,
    patient: str,
    *,
    lemma: str,
    past: str,
    polarity: str = "affirmed",
    modality: str = "asserted",
    adjunct: str = "",
) -> str:
    if polarity not in {"affirmed", "negated"}:
        raise ValueError(f"unsupported polarity: {polarity}")
    if modality not in {"asserted", "possible", "certain"}:
        raise ValueError(f"unsupported modality: {modality}")
    passive = _is_passive(template_id)
    if passive:
        if polarity == "negated":
            relation = f"was not {past} by"
        elif modality == "possible":
            relation = f"may have been {past} by"
        elif modality == "certain":
            relation = f"was definitely {past} by"
        else:
            relation = f"was {past} by"
        core = f"{patient} {relation} {actor}"
    else:
        if polarity == "negated":
            relation = f"did not {lemma}"
        elif modality == "possible":
            relation = f"may {lemma}"
        elif modality == "certain":
            relation = f"definitely {past}"
        else:
            relation = past
        core = f"{actor} {relation} {patient}"
    if adjunct:
        core = f"{core} {adjunct}"
    return _decorate(template_id, f"{core}.")


def _document(template_id: str, relevant: str, distractor: str) -> str:
    if template_id in {"active-leading", "passive-records"}:
        return f"{relevant} {distractor}"
    return f"{distractor} {relevant}"


def _annotation(
    axis: SemanticAxis,
    *,
    relevant: bool,
    changed: bool,
    before: str,
    after: str,
) -> SemanticVariableChange:
    return SemanticVariableChange(
        axis=axis,
        frame_id="target" if relevant else "distractor",
        query_relevant=relevant,
        value_changed=changed,
        before_value=before,
        after_value=after,
    )


def _intervention(
    base_text: str,
    *,
    target: str,
    replacement: str,
    template_id: str,
    predicate_family: str,
    axis: SemanticAxis,
    role: str,
    annotation: SemanticVariableChange,
) -> Intervention:
    annotation.validate_relation(annotation.expected_relation)
    return make_exact_replacement(
        base_text,
        target=target,
        replacement=replacement,
        family=axis.value,
        expected_relation=annotation.expected_relation,
        generator_id=(
            f"pilot-v4.{template_id}.{predicate_family}.{axis.value}.{role}"
        ),
        verification_status=VerificationStatus.AUTOMATIC,
        metadata={
            "pilot_contract": "synthetic-development-only",
            "template_id": template_id,
            "predicate_family": predicate_family,
            "adversarial_role": role,
            "pair_id": axis.value,
            "semantic_change": annotation.to_dict(),
        },
    )


def make_adversarial_frame_orbit(
    *,
    index: int,
    predicate: PredicateFamily,
    template_id: str,
    actor: str,
    patient: str,
    distractor_actor: str,
    distractor_patient: str,
    seed: int,
) -> InterventionOrbit:
    relevant = _clause(
        template_id,
        actor,
        patient,
        lemma=predicate.lemma,
        past=predicate.past,
    )
    distractor = _clause(
        template_id,
        distractor_actor,
        distractor_patient,
        lemma=predicate.lemma,
        past=predicate.past,
    )
    base_text = _document(template_id, relevant, distractor)
    query = f"{actor} {predicate.past} {patient}."

    relation_critical = _clause(
        template_id,
        actor,
        patient,
        lemma=predicate.contrast_lemma,
        past=predicate.contrast_past,
    )
    relation_control = _clause(
        template_id,
        distractor_actor,
        distractor_patient,
        lemma=predicate.contrast_lemma,
        past=predicate.contrast_past,
    )
    relation_invariant = _clause(
        template_id,
        actor,
        patient,
        lemma=predicate.paraphrase_lemma,
        past=predicate.paraphrase_past,
    )

    direction_critical = _clause(
        template_id,
        patient,
        actor,
        lemma=predicate.lemma,
        past=predicate.past,
    )
    direction_control = _clause(
        template_id,
        distractor_patient,
        distractor_actor,
        lemma=predicate.lemma,
        past=predicate.past,
    )
    alternate_voice = (
        template_id.replace("passive-", "active-", 1)
        if _is_passive(template_id)
        else template_id.replace("active-", "passive-", 1)
    )
    if alternate_voice not in TEMPLATE_IDS:
        alternate_voice = "passive-leading" if not _is_passive(template_id) else "active-leading"
    direction_invariant = _clause(
        alternate_voice,
        actor,
        patient,
        lemma=predicate.lemma,
        past=predicate.past,
    )

    scope_critical = _clause(
        template_id,
        actor,
        patient,
        lemma=predicate.lemma,
        past=predicate.past,
        polarity="negated",
    )
    scope_control = _clause(
        template_id,
        distractor_actor,
        distractor_patient,
        lemma=predicate.lemma,
        past=predicate.past,
        polarity="negated",
    )
    scope_invariant = _clause(
        template_id,
        actor,
        patient,
        lemma=predicate.lemma,
        past=predicate.past,
        adjunct="without delay",
    )

    modality_critical = _clause(
        template_id,
        actor,
        patient,
        lemma=predicate.lemma,
        past=predicate.past,
        modality="possible",
    )
    modality_control = _clause(
        template_id,
        distractor_actor,
        distractor_patient,
        lemma=predicate.lemma,
        past=predicate.past,
        modality="possible",
    )
    modality_invariant = _clause(
        template_id,
        actor,
        patient,
        lemma=predicate.lemma,
        past=predicate.past,
        modality="certain",
    )

    values = {
        SemanticAxis.RELATION: (
            (
                "critical", relevant, relation_critical, True, True,
                predicate.family_id, f"not-{predicate.family_id}",
            ),
            (
                "control", distractor, relation_control, False, True,
                predicate.family_id, f"not-{predicate.family_id}",
            ),
            (
                "invariant", relevant, relation_invariant, True, False,
                predicate.family_id, predicate.family_id,
            ),
        ),
        SemanticAxis.DIRECTION: (
            (
                "critical", relevant, direction_critical, True, True,
                f"{actor}>{patient}", f"{patient}>{actor}",
            ),
            (
                "control", distractor, direction_control, False, True,
                f"{distractor_actor}>{distractor_patient}",
                f"{distractor_patient}>{distractor_actor}",
            ),
            (
                "invariant", relevant, direction_invariant, True, False,
                f"{actor}>{patient}", f"{actor}>{patient}",
            ),
        ),
        SemanticAxis.SCOPE: (
            ("critical", relevant, scope_critical, True, True, "affirmed", "negated"),
            ("control", distractor, scope_control, False, True, "affirmed", "negated"),
            ("invariant", relevant, scope_invariant, True, False, "affirmed", "affirmed"),
        ),
        SemanticAxis.MODALITY: (
            ("critical", relevant, modality_critical, True, True, "asserted", "possible"),
            ("control", distractor, modality_control, False, True, "asserted", "possible"),
            ("invariant", relevant, modality_invariant, True, False, "asserted", "asserted"),
        ),
    }
    interventions = tuple(
        _intervention(
            base_text,
            target=target,
            replacement=replacement,
            template_id=template_id,
            predicate_family=predicate.family_id,
            axis=axis,
            role=role,
            annotation=_annotation(
                axis,
                relevant=relevant_flag,
                changed=changed,
                before=before,
                after=after,
            ),
        )
        for axis, edits in values.items()
        for role, target, replacement, relevant_flag, changed, before, after in edits
    )
    validate_unique_generator_ids(interventions)
    fingerprint = hashlib.sha256(
        (
            f"{predicate.family_id}:{template_id}:{actor}:{patient}:"
            f"{distractor_actor}:{distractor_patient}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return InterventionOrbit(
        base_id=f"pilot-v4-{index:05d}-{fingerprint}",
        base_text=base_text,
        context_text=query,
        interventions=interventions,
        source="synthetic-adversarial-frame-pilot-v4",
        metadata={
            "development_only": True,
            "claim_eligible": False,
            "language": "en",
            "template_id": template_id,
            "predicate_family": predicate.family_id,
            "evaluation_partition": evaluation_partition(
                predicate.family_id, template_id
            ),
            "seed": seed,
            "factor_fingerprint": fingerprint,
        },
    )


def generate_adversarial_frame_orbits(
    count: int, *, seed: int
) -> list[InterventionOrbit]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    used: set[tuple[str, str, str, str, str, str]] = set()
    orbits: list[InterventionOrbit] = []
    while len(orbits) < count:
        index = len(orbits)
        template_id = TEMPLATE_IDS[index % len(TEMPLATE_IDS)]
        predicate = PREDICATE_FAMILIES[
            (index // len(TEMPLATE_IDS)) % len(PREDICATE_FAMILIES)
        ]
        actor, distractor_actor = rng.sample(ACTORS, 2)
        patient, distractor_patient = rng.sample(PATIENTS, 2)
        key = (
            predicate.family_id,
            template_id,
            actor,
            patient,
            distractor_actor,
            distractor_patient,
        )
        if key in used:
            continue
        used.add(key)
        orbits.append(
            make_adversarial_frame_orbit(
                index=index,
                predicate=predicate,
                template_id=template_id,
                actor=actor,
                patient=patient,
                distractor_actor=distractor_actor,
                distractor_patient=distractor_patient,
                seed=seed,
            )
        )
    return orbits


def run(
    output: Path, *, count: int, seed: int, overwrite: bool
) -> tuple[Path, Path]:
    output = output.resolve()
    manifest = output.with_name(f"{output.name}.manifest.json")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    orbits = generate_adversarial_frame_orbits(count, seed=seed)
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
                "development_only": True,
                "claim_eligible": False,
                "human_authored": False,
                "generator": "synthetic-adversarial-frame-pilot-v4",
                "seed": seed,
                "orbit_count": len(orbits),
                "intervention_count": sum(len(x.interventions) for x in orbits),
                "template_counts": dict(
                    sorted(Counter(x.metadata["template_id"] for x in orbits).items())
                ),
                "predicate_family_counts": dict(
                    sorted(Counter(x.metadata["predicate_family"] for x in orbits).items())
                ),
                "partition_counts": dict(
                    sorted(Counter(x.metadata["evaluation_partition"] for x in orbits).items())
                ),
                "held_predicate_families": sorted(HELD_PREDICATE_FAMILIES),
                "held_template_ids": sorted(HELD_TEMPLATE_IDS),
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
        description="Generate the locked synthetic adversarial frame pilot v4."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=192)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output, manifest = run(
        args.output, count=args.count, seed=args.seed, overwrite=args.overwrite
    )
    print(f"wrote {output}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
