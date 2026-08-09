from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Sequence

from spectra_v3.interventions import (
    ExpectedRelation,
    Intervention,
    InterventionOrbit,
    VerificationStatus,
    make_exact_replacement,
    make_surface_intervention,
)

REPORTERS = (
    "Alice Morgan",
    "Bruno Silva",
    "Clara Jensen",
    "David Okafor",
    "Elena Rossi",
    "Farah Khan",
    "Gabriel Martin",
    "Hana Suzuki",
    "Ivan Petrov",
    "Julia Costa",
)
ORGANIZATIONS = (
    "Northstar Labs",
    "Meridian Works",
    "Aster Systems",
    "Blue Harbor Group",
    "Cedar Analytics",
    "Delta Foundry",
    "Evergreen Robotics",
    "Frontier Medical",
    "Granite Networks",
    "Helix Energy",
    "Ionis Transit",
    "Juniper Foods",
)
CITIES = (
    "Berlin",
    "Copenhagen",
    "Dublin",
    "Florence",
    "Geneva",
    "Helsinki",
    "Lisbon",
    "Oslo",
    "Prague",
    "Tallinn",
    "Vienna",
    "Warsaw",
)
QUANTITIES = (1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7500, 8000, 9000, 10000)
YEARS = tuple(range(2018, 2030))
MATCHED_REPORTERS = (
    "Alice",
    "Bruno",
    "Clara",
    "David",
    "Elena",
    "Gabriel",
    "Ivan",
    "Julia",
)
MATCHED_ORGANIZATIONS = (
    "Atlas",
    "Delta",
    "Cedar",
    "Summit",
    "Pioneer",
    "Liberty",
    "Phoenix",
    "Central",
    "Global",
    "Royal",
)
MATCHED_QUANTITIES = tuple(range(1000, 10000, 1000))
REPORT_IDS = tuple(range(1, 10))


def _replacement(
    base_text: str,
    *,
    target: str,
    replacement: str,
    family: str,
    relation: ExpectedRelation,
    generator_id: str,
    strength: float,
) -> Intervention:
    return make_exact_replacement(
        base_text,
        target=target,
        replacement=replacement,
        family=family,
        expected_relation=relation,
        generator_id=generator_id,
        strength=strength,
        verification_status=VerificationStatus.AUTOMATIC,
        metadata={"pilot_contract": "development_only"},
    )


def make_pilot_orbit(
    *,
    index: int,
    reporter_index: int,
    organization_index: int,
    city_index: int,
    quantity_index: int,
    year_index: int,
    seed: int,
) -> InterventionOrbit:
    reporter = REPORTERS[reporter_index]
    organization = ORGANIZATIONS[organization_index]
    city = CITIES[city_index]
    quantity = QUANTITIES[quantity_index]
    year = YEARS[year_index]
    alternate_reporter = REPORTERS[(reporter_index + 1) % len(REPORTERS)]
    alternate_organization = ORGANIZATIONS[
        (organization_index + 1) % len(ORGANIZATIONS)
    ]
    formatted_quantity = f"{quantity:,}"
    changed_quantity = f"{quantity + 500:,}"

    base_text = (
        f"According to analyst {reporter}, {organization} approved a shipment of "
        f"{formatted_quantity} units to {city} in {year}. The report is not only "
        "concise but also public, and the plan must pass a final inspection."
    )

    interventions = (
        make_surface_intervention(
            base_text,
            transformed_text=base_text.lower(),
            generator_id="pilot.surface.lowercase.v1",
            strength=0.05,
            metadata={"pilot_contract": "development_only"},
        ),
        make_surface_intervention(
            base_text,
            transformed_text=base_text.replace(". The", ".  The", 1),
            generator_id="pilot.surface.whitespace.v1",
            strength=0.05,
            metadata={"pilot_contract": "development_only"},
        ),
        _replacement(
            base_text,
            target="shipment",
            replacement="delivery",
            family="lexical",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot.lexical.shipment-delivery.v1",
            strength=0.25,
        ),
        _replacement(
            base_text,
            target="final inspection",
            replacement="last inspection",
            family="lexical",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot.lexical.final-last.v1",
            strength=0.25,
        ),
        _replacement(
            base_text,
            target=f"{organization} approved",
            replacement=f"{organization} did not approve",
            family="negation",
            relation=ExpectedRelation.CHANGE,
            generator_id="pilot.negation.approval.v1",
            strength=1.0,
        ),
        _replacement(
            base_text,
            target="not only concise but also public",
            replacement="concise and public",
            family="negation",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot.negation.not-only-control.v1",
            strength=0.2,
        ),
        _replacement(
            base_text,
            target=organization,
            replacement=alternate_organization,
            family="entity",
            relation=ExpectedRelation.CHANGE,
            generator_id="pilot.entity.organization.v1",
            strength=0.9,
        ),
        _replacement(
            base_text,
            target=reporter,
            replacement=alternate_reporter,
            family="entity",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot.entity.reporter-control.v1",
            strength=0.2,
        ),
        _replacement(
            base_text,
            target=formatted_quantity,
            replacement=changed_quantity,
            family="quantity",
            relation=ExpectedRelation.CHANGE,
            generator_id="pilot.quantity.value.v1",
            strength=0.8,
        ),
        _replacement(
            base_text,
            target=formatted_quantity,
            replacement=str(quantity),
            family="quantity",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot.quantity.format-control.v1",
            strength=0.1,
        ),
        _replacement(
            base_text,
            target=f"in {year}",
            replacement=f"in {year + 1}",
            family="time_modality",
            relation=ExpectedRelation.CHANGE,
            generator_id="pilot.time.year.v1",
            strength=0.8,
        ),
        _replacement(
            base_text,
            target="must pass",
            replacement="is required to pass",
            family="time_modality",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot.modality.required-control.v1",
            strength=0.2,
        ),
    )

    fingerprint = hashlib.sha256(
        f"{reporter_index}:{organization_index}:{city_index}:"
        f"{quantity_index}:{year_index}".encode("ascii")
    ).hexdigest()[:16]
    return InterventionOrbit(
        base_id=f"pilot-{index:05d}-{fingerprint}",
        base_text=base_text,
        interventions=interventions,
        source="synthetic-factorial-pilot-v1",
        metadata={
            "development_only": True,
            "language": "en",
            "template_id": "logistics-report-v1",
            "seed": seed,
            "factor_fingerprint": fingerprint,
        },
    )


def generate_pilot_orbits(count_: int, *, seed: int) -> list[InterventionOrbit]:
    if count_ <= 0:
        raise ValueError("count must be positive")
    maximum = (
        len(REPORTERS)
        * len(ORGANIZATIONS)
        * len(CITIES)
        * len(QUANTITIES)
        * len(YEARS)
    )
    if count_ > maximum:
        raise ValueError(f"count cannot exceed {maximum}")

    rng = random.Random(seed)
    factors: set[tuple[int, int, int, int, int]] = set()
    orbits: list[InterventionOrbit] = []
    for _ in count():
        key = (
            rng.randrange(len(REPORTERS)),
            rng.randrange(len(ORGANIZATIONS)),
            rng.randrange(len(CITIES)),
            rng.randrange(len(QUANTITIES)),
            rng.randrange(len(YEARS)),
        )
        if key in factors:
            continue
        factors.add(key)
        orbits.append(
            make_pilot_orbit(
                index=len(orbits),
                reporter_index=key[0],
                organization_index=key[1],
                city_index=key[2],
                quantity_index=key[3],
                year_index=key[4],
                seed=seed,
            )
        )
        if len(orbits) == count_:
            return orbits
    raise RuntimeError("unreachable pilot generation state")


def make_matched_pilot_orbit(
    *,
    index: int,
    reporter_index: int,
    organization_index: int,
    city_index: int,
    quantity_index: int,
    report_id_index: int,
    seed: int,
) -> InterventionOrbit:
    reporter = MATCHED_REPORTERS[reporter_index]
    organization = MATCHED_ORGANIZATIONS[organization_index]
    city = CITIES[city_index]
    quantity = MATCHED_QUANTITIES[quantity_index]
    report_id = REPORT_IDS[report_id_index]
    alternate_reporter = MATCHED_REPORTERS[
        (reporter_index + 1) % len(MATCHED_REPORTERS)
    ]
    alternate_organization = MATCHED_ORGANIZATIONS[
        (organization_index + 1) % len(MATCHED_ORGANIZATIONS)
    ]
    alternate_report_id = REPORT_IDS[(report_id_index + 1) % len(REPORT_IDS)]
    formatted_quantity = f"{quantity:,}"
    changed_quantity = f"{quantity + 500:,}"

    base_text = (
        f"Analyst {reporter} filed report {report_id}, which was approved for publication "
        f"in 2018. A shipment of {formatted_quantity} units to {city} was approved by "
        f"{organization} in 2025."
    )
    context_text = (
        f"A shipment of {formatted_quantity} units to {city} was approved by "
        f"{organization} in 2025."
    )
    metadata = {
        "pilot_contract": "development_only",
        "task_scope": "shipment_event",
        "matched_design": True,
    }
    interventions = (
        make_surface_intervention(
            base_text,
            transformed_text=base_text.lower(),
            generator_id="pilot-v2.surface.lowercase",
            strength=0.05,
            metadata=metadata,
        ),
        make_surface_intervention(
            base_text,
            transformed_text=base_text.replace(". A", ".  A", 1),
            generator_id="pilot-v2.surface.whitespace",
            strength=0.05,
            metadata=metadata,
        ),
        _replacement(
            base_text,
            target="shipment",
            replacement="delivery",
            family="lexical",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot-v2.lexical.shipment-delivery",
            strength=0.25,
        ),
        _replacement(
            base_text,
            target="filed report",
            replacement="submitted report",
            family="lexical",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot-v2.lexical.filed-submitted",
            strength=0.25,
        ),
        _replacement(
            base_text,
            target="was approved by",
            replacement="was not approved by",
            family="negation",
            relation=ExpectedRelation.CHANGE,
            generator_id="pilot-v2.negation.shipment",
            strength=1.0,
        ),
        _replacement(
            base_text,
            target="was approved for publication",
            replacement="was not approved for publication",
            family="negation",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot-v2.negation.report-control",
            strength=0.2,
        ),
        _replacement(
            base_text,
            target=organization,
            replacement=alternate_organization,
            family="entity",
            relation=ExpectedRelation.CHANGE,
            generator_id="pilot-v2.entity.organization",
            strength=0.9,
        ),
        _replacement(
            base_text,
            target=reporter,
            replacement=alternate_reporter,
            family="entity",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot-v2.entity.reporter-control",
            strength=0.2,
        ),
        _replacement(
            base_text,
            target=formatted_quantity,
            replacement=changed_quantity,
            family="quantity",
            relation=ExpectedRelation.CHANGE,
            generator_id="pilot-v2.quantity.shipment",
            strength=0.8,
        ),
        _replacement(
            base_text,
            target=f"report {report_id}",
            replacement=f"report {alternate_report_id}",
            family="quantity",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot-v2.quantity.report-id-control",
            strength=0.1,
        ),
        _replacement(
            base_text,
            target="in 2025",
            replacement="in 2026",
            family="time_modality",
            relation=ExpectedRelation.CHANGE,
            generator_id="pilot-v2.time.shipment-year",
            strength=0.8,
        ),
        _replacement(
            base_text,
            target="in 2018",
            replacement="in 2019",
            family="time_modality",
            relation=ExpectedRelation.PRESERVE,
            generator_id="pilot-v2.time.publication-year-control",
            strength=0.2,
        ),
    )
    interventions = tuple(
        Intervention(
            **{
                **item.to_dict(),
                "expected_relation": item.expected_relation,
                "verification_status": item.verification_status,
                "metadata": {**item.metadata, **metadata},
            }
        )
        for item in interventions
    )

    fingerprint = hashlib.sha256(
        f"{reporter_index}:{organization_index}:{city_index}:"
        f"{quantity_index}:{report_id_index}".encode("ascii")
    ).hexdigest()[:16]
    return InterventionOrbit(
        base_id=f"pilot-v2-{index:05d}-{fingerprint}",
        base_text=base_text,
        interventions=interventions,
        source="synthetic-token-matched-pilot-v2",
        context_text=context_text,
        metadata={
            "development_only": True,
            "language": "en",
            "template_id": "parallel-report-shipment-v2",
            "task_scope": "shipment_event",
            "matched_for_encoder_revision": (
                "sentence-transformers/all-MiniLM-L6-v2@"
                "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
            ),
            "seed": seed,
            "factor_fingerprint": fingerprint,
        },
    )


def generate_matched_pilot_orbits(
    count_: int, *, seed: int
) -> list[InterventionOrbit]:
    if count_ <= 0:
        raise ValueError("count must be positive")
    maximum = (
        len(MATCHED_REPORTERS)
        * len(MATCHED_ORGANIZATIONS)
        * len(CITIES)
        * len(MATCHED_QUANTITIES)
        * len(REPORT_IDS)
    )
    if count_ > maximum:
        raise ValueError(f"count cannot exceed {maximum}")

    rng = random.Random(seed)
    factors: set[tuple[int, int, int, int, int]] = set()
    orbits: list[InterventionOrbit] = []
    for _ in count():
        key = (
            rng.randrange(len(MATCHED_REPORTERS)),
            rng.randrange(len(MATCHED_ORGANIZATIONS)),
            rng.randrange(len(CITIES)),
            rng.randrange(len(MATCHED_QUANTITIES)),
            rng.randrange(len(REPORT_IDS)),
        )
        if key in factors:
            continue
        factors.add(key)
        orbits.append(
            make_matched_pilot_orbit(
                index=len(orbits),
                reporter_index=key[0],
                organization_index=key[1],
                city_index=key[2],
                quantity_index=key[3],
                report_id_index=key[4],
                seed=seed,
            )
        )
        if len(orbits) == count_:
            return orbits
    raise RuntimeError("unreachable matched pilot generation state")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic Phase 0 pilot corpus.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--design", choices=("v1", "v2"), default="v2")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(
    output: Path, *, count_: int, seed: int, design: str, overwrite: bool
) -> tuple[Path, Path]:
    output = output.resolve()
    manifest = output.with_name(f"{output.name}.manifest.json")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if design == "v1":
        orbits = generate_pilot_orbits(count_, seed=seed)
        generator = "synthetic-factorial-pilot-v1"
    elif design == "v2":
        orbits = generate_matched_pilot_orbits(count_, seed=seed)
        generator = "synthetic-token-matched-pilot-v2"
    else:
        raise ValueError("design must be 'v1' or 'v2'")

    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for orbit in orbits:
            record = json.dumps(orbit.to_dict(), ensure_ascii=False, sort_keys=True)
            handle.write(record + "\n")
    temporary.replace(output)

    family_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    for orbit in orbits:
        family_counts.update(item.family for item in orbit.interventions)
        relation_counts.update(item.expected_relation.value for item in orbit.interventions)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "development_only": True,
                "claim_eligible": False,
                "generator": generator,
                "seed": seed,
                "orbit_count": len(orbits),
                "intervention_count": sum(len(orbit.interventions) for orbit in orbits),
                "family_counts": dict(sorted(family_counts.items())),
                "relation_counts": dict(sorted(relation_counts.items())),
                "sha256": digest,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output, manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output, manifest = run(
        args.output,
        count_=args.count,
        seed=args.seed,
        design=args.design,
        overwrite=args.overwrite,
    )
    print(f"wrote {output}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
