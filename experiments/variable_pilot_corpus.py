from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from experiments.pilot_corpus import (
    CITIES,
    MATCHED_ORGANIZATIONS,
    MATCHED_QUANTITIES,
    MATCHED_REPORTERS,
    REPORT_IDS,
)
from spectra_v3.interventions import (
    Intervention,
    InterventionOrbit,
    VerificationStatus,
    make_exact_replacement,
)
from spectra_v3.semantic_variables import SemanticAxis, SemanticVariableChange


@dataclass(frozen=True)
class ChangeSpec:
    target: str
    replacement: str
    frame_id: str
    query_relevant: bool
    before_value: str
    after_value: str
    value_changed: bool = True


@dataclass(frozen=True)
class TemplateInstance:
    template_id: str
    base_text: str
    context_text: str
    changes: dict[SemanticAxis, tuple[ChangeSpec, ...]]


def _values(
    reporter_index: int,
    organization_index: int,
    city_index: int,
    quantity_index: int,
    report_id_index: int,
) -> dict[str, str]:
    reporter = MATCHED_REPORTERS[reporter_index]
    organization = MATCHED_ORGANIZATIONS[organization_index]
    report_id = REPORT_IDS[report_id_index]
    quantity = MATCHED_QUANTITIES[quantity_index]
    return {
        "reporter": reporter,
        "alternate_reporter": MATCHED_REPORTERS[
            (reporter_index + 1) % len(MATCHED_REPORTERS)
        ],
        "organization": organization,
        "alternate_organization": MATCHED_ORGANIZATIONS[
            (organization_index + 1) % len(MATCHED_ORGANIZATIONS)
        ],
        "city": CITIES[city_index],
        "quantity": f"{quantity:,}",
        "alternate_quantity": f"{quantity + 500:,}",
        "report_id": str(report_id),
        "alternate_report_id": str(
            REPORT_IDS[(report_id_index + 1) % len(REPORT_IDS)]
        ),
    }


def _template_parallel(value: dict[str, str]) -> TemplateInstance:
    base = (
        f"Analyst {value['reporter']} filed report {value['report_id']}, which was "
        "approved for publication in 2018. A shipment of "
        f"{value['quantity']} units to {value['city']} was approved by "
        f"{value['organization']} in 2025."
    )
    context = (
        f"A shipment of {value['quantity']} units to {value['city']} was approved by "
        f"{value['organization']} in 2025."
    )
    return TemplateInstance(
        template_id="parallel",
        base_text=base,
        context_text=context,
        changes={
            SemanticAxis.RELATION: (
                ChangeSpec(
                    "was approved by",
                    "was rejected by",
                    "shipment",
                    True,
                    "approved",
                    "rejected",
                ),
                ChangeSpec(
                    "approved for publication",
                    "rejected for publication",
                    "report",
                    False,
                    "approved",
                    "rejected",
                ),
                ChangeSpec(
                    "shipment",
                    "delivery",
                    "shipment",
                    True,
                    "transport_event",
                    "transport_event",
                    False,
                ),
            ),
            SemanticAxis.POLARITY: (
                ChangeSpec(
                    "was approved by",
                    "was not approved by",
                    "shipment",
                    True,
                    "affirmed",
                    "negated",
                ),
                ChangeSpec(
                    "was approved for publication",
                    "was not approved for publication",
                    "report",
                    False,
                    "affirmed",
                    "negated",
                ),
            ),
            SemanticAxis.ENTITY: (
                ChangeSpec(
                    value["organization"],
                    value["alternate_organization"],
                    "shipment",
                    True,
                    value["organization"],
                    value["alternate_organization"],
                ),
                ChangeSpec(
                    value["reporter"],
                    value["alternate_reporter"],
                    "report",
                    False,
                    value["reporter"],
                    value["alternate_reporter"],
                ),
            ),
            SemanticAxis.QUANTITY: (
                ChangeSpec(
                    value["quantity"],
                    value["alternate_quantity"],
                    "shipment",
                    True,
                    value["quantity"],
                    value["alternate_quantity"],
                ),
                ChangeSpec(
                    f"report {value['report_id']}",
                    f"report {value['alternate_report_id']}",
                    "report",
                    False,
                    value["report_id"],
                    value["alternate_report_id"],
                ),
            ),
            SemanticAxis.TIME: (
                ChangeSpec("in 2025", "in 2026", "shipment", True, "2025", "2026"),
                ChangeSpec("in 2018", "in 2019", "report", False, "2018", "2019"),
            ),
        },
    )


def _template_leading_event(value: dict[str, str]) -> TemplateInstance:
    base = (
        f"In 2025, {value['organization']} approved a shipment of {value['quantity']} "
        f"units to {value['city']}. In 2018, analyst {value['reporter']} approved "
        f"report {value['report_id']} for publication."
    )
    context = (
        f"In 2025, {value['organization']} approved a shipment of {value['quantity']} "
        f"units to {value['city']}."
    )
    return TemplateInstance(
        template_id="leading-event",
        base_text=base,
        context_text=context,
        changes={
            SemanticAxis.RELATION: (
                ChangeSpec(
                    f"{value['organization']} approved",
                    f"{value['organization']} rejected",
                    "shipment",
                    True,
                    "approved",
                    "rejected",
                ),
                ChangeSpec(
                    f"{value['reporter']} approved",
                    f"{value['reporter']} rejected",
                    "report",
                    False,
                    "approved",
                    "rejected",
                ),
                ChangeSpec(
                    "shipment",
                    "delivery",
                    "shipment",
                    True,
                    "transport_event",
                    "transport_event",
                    False,
                ),
            ),
            SemanticAxis.POLARITY: (
                ChangeSpec(
                    f"{value['organization']} approved",
                    f"{value['organization']} did not approve",
                    "shipment",
                    True,
                    "affirmed",
                    "negated",
                ),
                ChangeSpec(
                    f"{value['reporter']} approved",
                    f"{value['reporter']} did not approve",
                    "report",
                    False,
                    "affirmed",
                    "negated",
                ),
            ),
            SemanticAxis.ENTITY: (
                ChangeSpec(
                    value["organization"],
                    value["alternate_organization"],
                    "shipment",
                    True,
                    value["organization"],
                    value["alternate_organization"],
                ),
                ChangeSpec(
                    value["reporter"],
                    value["alternate_reporter"],
                    "report",
                    False,
                    value["reporter"],
                    value["alternate_reporter"],
                ),
            ),
            SemanticAxis.QUANTITY: (
                ChangeSpec(
                    value["quantity"],
                    value["alternate_quantity"],
                    "shipment",
                    True,
                    value["quantity"],
                    value["alternate_quantity"],
                ),
                ChangeSpec(
                    f"report {value['report_id']}",
                    f"report {value['alternate_report_id']}",
                    "report",
                    False,
                    value["report_id"],
                    value["alternate_report_id"],
                ),
            ),
            SemanticAxis.TIME: (
                ChangeSpec("In 2025", "In 2026", "shipment", True, "2025", "2026"),
                ChangeSpec("In 2018", "In 2019", "report", False, "2018", "2019"),
            ),
        },
    )


def _template_records(value: dict[str, str]) -> TemplateInstance:
    base = (
        f"A 2018 publication note records that {value['reporter']} authorized report "
        f"{value['report_id']}. A 2025 logistics note records that "
        f"{value['organization']} authorized {value['quantity']} units for "
        f"{value['city']}."
    )
    context = (
        f"In 2025, {value['organization']} authorized {value['quantity']} units for "
        f"{value['city']}."
    )
    return TemplateInstance(
        template_id="records",
        base_text=base,
        context_text=context,
        changes={
            SemanticAxis.RELATION: (
                ChangeSpec(
                    f"{value['organization']} authorized",
                    f"{value['organization']} allocated",
                    "shipment",
                    True,
                    "authorized",
                    "allocated",
                ),
                ChangeSpec(
                    f"{value['reporter']} authorized",
                    f"{value['reporter']} allocated",
                    "report",
                    False,
                    "authorized",
                    "allocated",
                ),
                ChangeSpec(
                    f"{value['organization']} authorized",
                    f"{value['organization']} approved",
                    "shipment",
                    True,
                    "permission_granted",
                    "permission_granted",
                    False,
                ),
            ),
            SemanticAxis.POLARITY: (
                ChangeSpec(
                    f"{value['organization']} authorized",
                    f"{value['organization']} did not authorize",
                    "shipment",
                    True,
                    "affirmed",
                    "negated",
                ),
                ChangeSpec(
                    f"{value['reporter']} authorized",
                    f"{value['reporter']} did not authorize",
                    "report",
                    False,
                    "affirmed",
                    "negated",
                ),
            ),
            SemanticAxis.ENTITY: (
                ChangeSpec(
                    value["organization"],
                    value["alternate_organization"],
                    "shipment",
                    True,
                    value["organization"],
                    value["alternate_organization"],
                ),
                ChangeSpec(
                    value["reporter"],
                    value["alternate_reporter"],
                    "report",
                    False,
                    value["reporter"],
                    value["alternate_reporter"],
                ),
            ),
            SemanticAxis.QUANTITY: (
                ChangeSpec(
                    value["quantity"],
                    value["alternate_quantity"],
                    "shipment",
                    True,
                    value["quantity"],
                    value["alternate_quantity"],
                ),
                ChangeSpec(
                    f"report {value['report_id']}",
                    f"report {value['alternate_report_id']}",
                    "report",
                    False,
                    value["report_id"],
                    value["alternate_report_id"],
                ),
            ),
            SemanticAxis.TIME: (
                ChangeSpec(
                    "A 2025 logistics", "A 2026 logistics", "shipment", True, "2025", "2026"
                ),
                ChangeSpec(
                    "A 2018 publication", "A 2019 publication", "report", False, "2018", "2019"
                ),
            ),
        },
    )


def _template_telegraphic(value: dict[str, str]) -> TemplateInstance:
    base = (
        f"Report {value['report_id']} from {value['reporter']} was accepted in 2018. "
        f"Shipment by {value['organization']} containing {value['quantity']} units for "
        f"{value['city']} was accepted in 2025."
    )
    context = (
        f"The shipment by {value['organization']} containing {value['quantity']} units "
        f"for {value['city']} was accepted in 2025."
    )
    return TemplateInstance(
        template_id="telegraphic",
        base_text=base,
        context_text=context,
        changes={
            SemanticAxis.RELATION: (
                ChangeSpec(
                    f"{value['city']} was accepted",
                    f"{value['city']} was rejected",
                    "shipment",
                    True,
                    "accepted",
                    "rejected",
                ),
                ChangeSpec(
                    f"{value['reporter']} was accepted",
                    f"{value['reporter']} was rejected",
                    "report",
                    False,
                    "accepted",
                    "rejected",
                ),
                ChangeSpec(
                    "Shipment",
                    "Delivery",
                    "shipment",
                    True,
                    "transport_event",
                    "transport_event",
                    False,
                ),
            ),
            SemanticAxis.POLARITY: (
                ChangeSpec(
                    f"{value['city']} was accepted",
                    f"{value['city']} was not accepted",
                    "shipment",
                    True,
                    "affirmed",
                    "negated",
                ),
                ChangeSpec(
                    f"{value['reporter']} was accepted",
                    f"{value['reporter']} was not accepted",
                    "report",
                    False,
                    "affirmed",
                    "negated",
                ),
            ),
            SemanticAxis.ENTITY: (
                ChangeSpec(
                    value["organization"],
                    value["alternate_organization"],
                    "shipment",
                    True,
                    value["organization"],
                    value["alternate_organization"],
                ),
                ChangeSpec(
                    value["reporter"],
                    value["alternate_reporter"],
                    "report",
                    False,
                    value["reporter"],
                    value["alternate_reporter"],
                ),
            ),
            SemanticAxis.QUANTITY: (
                ChangeSpec(
                    value["quantity"],
                    value["alternate_quantity"],
                    "shipment",
                    True,
                    value["quantity"],
                    value["alternate_quantity"],
                ),
                ChangeSpec(
                    f"Report {value['report_id']}",
                    f"Report {value['alternate_report_id']}",
                    "report",
                    False,
                    value["report_id"],
                    value["alternate_report_id"],
                ),
            ),
            SemanticAxis.TIME: (
                ChangeSpec("in 2025", "in 2026", "shipment", True, "2025", "2026"),
                ChangeSpec("in 2018", "in 2019", "report", False, "2018", "2019"),
            ),
        },
    )


TEMPLATES: tuple[Callable[[dict[str, str]], TemplateInstance], ...] = (
    _template_parallel,
    _template_leading_event,
    _template_records,
    _template_telegraphic,
)


def _make_intervention(
    instance: TemplateInstance,
    axis: SemanticAxis,
    spec: ChangeSpec,
) -> Intervention:
    annotation = SemanticVariableChange(
        axis=axis,
        frame_id=spec.frame_id,
        query_relevant=spec.query_relevant,
        value_changed=spec.value_changed,
        before_value=spec.before_value,
        after_value=spec.after_value,
    )
    if annotation.expected_relation.value == "change":
        role = "critical"
    elif spec.query_relevant:
        role = "invariant"
    else:
        role = "control"
    return make_exact_replacement(
        instance.base_text,
        target=spec.target,
        replacement=spec.replacement,
        family=axis.value,
        expected_relation=annotation.expected_relation,
        generator_id=f"pilot-v3.{instance.template_id}.{axis.value}.{role}",
        strength=1.0,
        verification_status=VerificationStatus.AUTOMATIC,
        metadata={
            "pilot_contract": "development_only",
            "template_id": instance.template_id,
            "semantic_change": annotation.to_dict(),
        },
    )


def make_variable_pilot_orbit(
    *,
    index: int,
    template_index: int,
    reporter_index: int,
    organization_index: int,
    city_index: int,
    quantity_index: int,
    report_id_index: int,
    seed: int,
) -> InterventionOrbit:
    value = _values(
        reporter_index,
        organization_index,
        city_index,
        quantity_index,
        report_id_index,
    )
    instance = TEMPLATES[template_index](value)
    interventions = tuple(
        _make_intervention(instance, axis, spec)
        for axis in (
            SemanticAxis.ENTITY,
            SemanticAxis.POLARITY,
            SemanticAxis.QUANTITY,
            SemanticAxis.RELATION,
            SemanticAxis.TIME,
        )
        for spec in instance.changes[axis]
    )
    fingerprint = hashlib.sha256(
        (
            f"{template_index}:{reporter_index}:{organization_index}:"
            f"{city_index}:{quantity_index}:{report_id_index}"
        ).encode("ascii")
    ).hexdigest()[:16]
    return InterventionOrbit(
        base_id=f"pilot-v3-{index:05d}-{fingerprint}",
        base_text=instance.base_text,
        context_text=instance.context_text,
        interventions=interventions,
        source="synthetic-semantic-variable-pilot-v3",
        metadata={
            "development_only": True,
            "language": "en",
            "template_id": instance.template_id,
            "semantic_variable_design": True,
            "seed": seed,
            "factor_fingerprint": fingerprint,
        },
    )


def generate_variable_pilot_orbits(
    count: int, *, seed: int
) -> list[InterventionOrbit]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    factors: set[tuple[int, int, int, int, int, int]] = set()
    orbits: list[InterventionOrbit] = []
    while len(orbits) < count:
        template_index = len(orbits) % len(TEMPLATES)
        key = (
            template_index,
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
            make_variable_pilot_orbit(
                index=len(orbits),
                template_index=key[0],
                reporter_index=key[1],
                organization_index=key[2],
                city_index=key[3],
                quantity_index=key[4],
                report_id_index=key[5],
                seed=seed,
            )
        )
    return orbits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the multi-template semantic-variable Phase 0 pilot."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(
    output: Path, *, count: int, seed: int, overwrite: bool
) -> tuple[Path, Path]:
    output = output.resolve()
    manifest = output.with_name(f"{output.name}.manifest.json")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    orbits = generate_variable_pilot_orbits(count, seed=seed)

    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for orbit in orbits:
            handle.write(json.dumps(orbit.to_dict(), sort_keys=True) + "\n")
    temporary.replace(output)

    template_counts = Counter(orbit.metadata["template_id"] for orbit in orbits)
    axis_counts = Counter(
        item.family for orbit in orbits for item in orbit.interventions
    )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "development_only": True,
                "claim_eligible": False,
                "generator": "synthetic-semantic-variable-pilot-v3",
                "seed": seed,
                "orbit_count": len(orbits),
                "intervention_count": sum(len(x.interventions) for x in orbits),
                "template_counts": dict(sorted(template_counts.items())),
                "axis_counts": dict(sorted(axis_counts.items())),
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output, manifest = run(
        args.output,
        count=args.count,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"wrote {output}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
