import json

from experiments.adversarial_frame_corpus import (
    HELD_PREDICATE_FAMILIES,
    HELD_TEMPLATE_IDS,
    generate_adversarial_frame_orbits,
    run,
)
from spectra_v3.semantic_variables import semantic_change_from_metadata


def test_v4_corpus_balances_axes_roles_and_locked_partitions() -> None:
    orbits = generate_adversarial_frame_orbits(32, seed=7)

    assert len({orbit.metadata["predicate_family"] for orbit in orbits}) == 8
    assert len({orbit.metadata["template_id"] for orbit in orbits}) == 4
    assert {orbit.metadata["evaluation_partition"] for orbit in orbits} == {
        "development",
        "held-predicate",
        "held-template",
        "double-holdout",
    }
    for orbit in orbits:
        assert len(orbit.interventions) == 12
        grouped = {}
        for item in orbit.interventions:
            change = semantic_change_from_metadata(item.metadata)
            change.validate_relation(item.expected_relation)
            grouped.setdefault(change.axis.value, set()).add(
                item.metadata["adversarial_role"]
            )
        assert grouped == {
            "direction": {"control", "critical", "invariant"},
            "modality": {"control", "critical", "invariant"},
            "relation": {"control", "critical", "invariant"},
            "scope": {"control", "critical", "invariant"},
        }


def test_v4_manifest_discloses_synthetic_holdouts(tmp_path) -> None:
    output, manifest = run(
        tmp_path / "orbits.jsonl", count=32, seed=11, overwrite=False
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert output.is_file()
    assert payload["development_only"] is True
    assert payload["claim_eligible"] is False
    assert payload["human_authored"] is False
    assert payload["intervention_count"] == 384
    assert set(payload["held_predicate_families"]) == HELD_PREDICATE_FAMILIES
    assert set(payload["held_template_ids"]) == HELD_TEMPLATE_IDS
