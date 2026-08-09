import json

from experiments.variable_pilot_corpus import generate_variable_pilot_orbits, run
from spectra_v3.semantic_variables import semantic_change_from_metadata


def test_variable_pilot_balances_templates_and_semantic_axes() -> None:
    orbits = generate_variable_pilot_orbits(8, seed=7)

    templates = [orbit.metadata["template_id"] for orbit in orbits]
    assert set(templates) == {"leading-event", "parallel", "records", "telegraphic"}
    assert all(templates.count(template) == 2 for template in set(templates))
    for orbit in orbits:
        assert len(orbit.interventions) == 11
        grouped = {}
        for item in orbit.interventions:
            annotation = semantic_change_from_metadata(item.metadata)
            annotation.validate_relation(item.expected_relation)
            grouped.setdefault(annotation.axis.value, set()).add(
                annotation.query_relevant
            )
        assert grouped == {
            "entity": {False, True},
            "polarity": {False, True},
            "quantity": {False, True},
            "relation": {False, True},
            "time": {False, True},
        }


def test_variable_pilot_run_writes_development_manifest(tmp_path) -> None:
    output, manifest = run(
        tmp_path / "orbits.jsonl",
        count=4,
        seed=11,
        overwrite=False,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert output.is_file()
    assert payload["development_only"] is True
    assert payload["claim_eligible"] is False
    assert payload["orbit_count"] == 4
    assert payload["intervention_count"] == 44
