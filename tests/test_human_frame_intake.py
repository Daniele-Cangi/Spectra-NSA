from dataclasses import replace
import json

import pytest

from experiments.adversarial_frame_corpus import generate_adversarial_frame_orbits
from experiments.human_frame_intake import HUMAN_SOURCE, run, validate_human_orbits
from spectra_v3.interventions import VerificationStatus
from spectra_v3.semantic_variables import semantic_change_from_metadata


AUTHOR = "sha256:" + "a" * 64
REVIEWER_A = "sha256:" + "b" * 64
REVIEWER_B = "sha256:" + "c" * 64
MAPPING_HASH = "d" * 64


def _human_orbit(axis: str = "relation"):
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
        if semantic_change_from_metadata(item.metadata).axis.value == axis
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
            "target_axis": axis,
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


def test_human_intake_locks_balanced_axis_local_corpus(tmp_path) -> None:
    axes = ("relation", "direction", "scope", "modality")
    authors = tuple("sha256:" + character * 64 for character in "def")
    source_groups = ("institutional", "dialogue", "narrative")
    orbits = []
    for index in range(96):
        axis = axes[index % len(axes)]
        orbit = _human_orbit(axis)
        author_index = index % len(authors)
        orbits.append(
            replace(
                orbit,
                base_id=f"human-case-{index:03d}",
                metadata={
                    **orbit.metadata,
                    "author_id_hash": authors[author_index],
                    "source_group": source_groups[author_index],
                },
            )
        )

    input_path = tmp_path / "reviewed.jsonl"
    input_path.write_text(
        "".join(json.dumps(orbit.to_dict()) + "\n" for orbit in orbits),
        encoding="utf-8",
    )
    output_path, manifest_path = run(
        input_path,
        tmp_path / "locked.jsonl",
        protocol_version="frame-v4.1",
        evaluation_commit="2" * 40,
        overwrite=False,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert output_path.is_file()
    assert manifest["orbit_count"] == 96
    assert manifest["intervention_count"] == 288
    assert manifest["author_count"] == 3
    assert sorted(manifest["author_counts"].values()) == [32, 32, 32]
    assert manifest["source_group_counts"] == {
        "dialogue": 32,
        "institutional": 32,
        "narrative": 32,
    }
    assert manifest["axis_counts"] == {
        "direction": 24,
        "modality": 24,
        "relation": 24,
        "scope": 24,
    }
