import numpy as np

from experiments.adversarial_frame_corpus import generate_adversarial_frame_orbits
from experiments.phase0_frame_observer import _prepare_frames, observe_orbits


def _exact(left: str, right: str) -> float:
    return float(left.casefold() == right.casefold())


def test_observer_separates_argument_set_from_direction() -> None:
    orbit = generate_adversarial_frame_orbits(1, seed=7)[0]
    row = observe_orbits([orbit], _prepare_frames([orbit]), _exact)[0]
    direction = {
        item["generator_id"].rsplit(".", 1)[-1]: item
        for item in row["interventions"]
        if item["family"] == "direction"
    }

    critical = direction["critical"]
    assert critical["frame_coordinate_drops"]["argument_alignment_drop"] == 0.0
    assert critical["frame_coordinate_drops"]["direction_alignment_drop"] == 1.0
    assert critical["frame_distance"] == 1.0
    assert direction["control"]["frame_distance"] == 0.0
    assert direction["invariant"]["frame_distance"] == 0.0


def test_observer_routes_scope_and_modality_without_design_fields() -> None:
    orbit = generate_adversarial_frame_orbits(1, seed=9)[0]
    row = observe_orbits([orbit], _prepare_frames([orbit]), _exact)[0]
    by_role = {
        (item["family"], item["generator_id"].rsplit(".", 1)[-1]): item
        for item in row["interventions"]
    }

    assert by_role[("scope", "critical")]["frame_distance"] == 1.0
    assert by_role[("scope", "invariant")]["frame_distance"] == 0.0
    assert by_role[("modality", "critical")]["frame_distance"] == 1.0
    assert by_role[("modality", "invariant")]["frame_distance"] == 0.0
    assert "adversarial_role" not in by_role[("scope", "critical")]


def test_embedding_lookup_contract_can_be_reproduced_with_unit_vectors() -> None:
    orbit = generate_adversarial_frame_orbits(1, seed=3)[0]
    prepared = _prepare_frames([orbit])
    texts = list(
        dict.fromkeys(
            text
            for frame in (prepared[0][0], prepared[0][1], *prepared[0][2])
            for text in (frame.span, frame.predicate, frame.actor, frame.patient)
            if text
        )
    )
    embeddings = np.eye(len(texts), dtype=np.float32)
    assert embeddings.shape[0] == len(texts)
