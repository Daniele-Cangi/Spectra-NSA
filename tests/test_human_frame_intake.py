from dataclasses import replace

import pytest

from experiments.adversarial_frame_corpus import generate_adversarial_frame_orbits
from experiments.human_frame_intake import HUMAN_SOURCE, validate_human_orbits
from spectra_v3.interventions import VerificationStatus


AUTHOR = "sha256:" + "a" * 64
REVIEWER_A = "sha256:" + "b" * 64
REVIEWER_B = "sha256:" + "c" * 64
MAPPING_HASH = "d" * 64


def _human_orbit():
    orbit = generate_adversarial_frame_orbits(1, seed=7)[0]
    interventions = tuple(
        replace(
            item,
            verification_status=VerificationStatus.HUMAN,
            metadata={
                **item.metadata,
                "human_annotation_id": f"annotation-{index}",
                "blinded_review": True,
                "reviewer_count": 2,
                "reviewer_id_hashes": [REVIEWER_A, REVIEWER_B],
                "review_protocol": "blind-frame-v1",
                "private_mapping_sha256": MAPPING_HASH,
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
            "author_id_hash": AUTHOR,
            "source_group": "batch-a",
            "collection_protocol": "frame-v4.1",
            "review_protocol": "blind-frame-v1",
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
