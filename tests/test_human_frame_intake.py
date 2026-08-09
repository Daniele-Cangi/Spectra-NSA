from dataclasses import replace

import pytest

from experiments.adversarial_frame_corpus import generate_adversarial_frame_orbits
from experiments.human_frame_intake import HUMAN_SOURCE, validate_human_orbits
from spectra_v3.interventions import VerificationStatus


def _human_orbit():
    orbit = generate_adversarial_frame_orbits(1, seed=7)[0]
    interventions = tuple(
        replace(
            item,
            verification_status=VerificationStatus.HUMAN,
            metadata={
                **item.metadata,
                "human_annotation_id": f"annotation-{index}",
            },
        )
        for index, item in enumerate(orbit.interventions)
    )
    return replace(
        orbit,
        source=HUMAN_SOURCE,
        interventions=interventions,
        metadata={
            **orbit.metadata,
            "human_authored": True,
            "evaluation_partition": "human-locked",
            "author_id_hash": "sha256:author",
            "source_group": "batch-a",
            "collection_protocol": "frame-v4.1",
        },
    )


def test_human_intake_accepts_complete_provenance_and_matched_roles() -> None:
    validate_human_orbits([_human_orbit()])


def test_human_intake_rejects_automatic_verification() -> None:
    orbit = _human_orbit()
    bad = replace(
        orbit,
        interventions=(
            replace(
                orbit.interventions[0],
                verification_status=VerificationStatus.AUTOMATIC,
            ),
            *orbit.interventions[1:],
        ),
    )
    with pytest.raises(ValueError, match="not human verified"):
        validate_human_orbits([bad])
