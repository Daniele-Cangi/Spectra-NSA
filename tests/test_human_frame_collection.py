import hashlib
import json

import pytest

from experiments.adversarial_frame_corpus import generate_adversarial_frame_orbits
from experiments.human_frame_collection import (
    compile_run,
    compile_reviewed_orbits,
    load_drafts,
    make_blind_review_packet,
    make_dry_run_kit,
)
from experiments.human_frame_intake import run as intake_run
from experiments.human_frame_intake import validate_human_orbits
from spectra_v3.semantic_variables import semantic_change_from_metadata


PROTOCOL = "human-frame-v1"
AUTHOR = "sha256:" + "a" * 64
REVIEWER_A = "sha256:" + "b" * 64
REVIEWER_B = "sha256:" + "c" * 64


def _draft_row(axis: str = "relation") -> dict:
    orbit = generate_adversarial_frame_orbits(1, seed=7)[0]
    items = []
    for item in orbit.interventions:
        change = semantic_change_from_metadata(item.metadata)
        if change.axis.value != axis:
            continue
        role = item.metadata["adversarial_role"]
        items.append(
            {
                "annotation_id": f"case-a-{change.axis.value}-{role}",
                "axis": change.axis.value,
                "role": role,
                "transformed_text": item.transformed_text,
                "frame_id": change.frame_id,
                "query_relevant": change.query_relevant,
                "value_changed": change.value_changed,
                "before_value": change.before_value,
                "after_value": change.after_value,
            }
        )
    return {
        "schema_version": 1,
        "draft_status": "complete",
        "case_id": "case-a",
        "author_id_hash": AUTHOR,
        "source_group": "source-a",
        "collection_protocol": PROTOCOL,
        "language": "en",
        "context_text": orbit.context_text,
        "base_text": orbit.base_text,
        "template_id": "free-clause",
        "predicate_family": "permission",
        "target_axis": axis,
        "items": items,
    }


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _packet(tmp_path):
    drafts_path = tmp_path / "drafts.jsonl"
    packet_path = tmp_path / "packet.jsonl"
    mapping_path = tmp_path / "private-map.jsonl"
    _write_jsonl(drafts_path, [_draft_row()])
    make_blind_review_packet(
        drafts_path,
        packet_path,
        mapping_path,
        protocol_version=PROTOCOL,
        seed=3,
        overwrite=False,
    )
    drafts = load_drafts(drafts_path, protocol_version=PROTOCOL)
    mapping = [
        json.loads(line)
        for line in mapping_path.read_text(encoding="utf-8").splitlines()
    ]
    return drafts, packet_path, mapping_path, mapping


def _reviews(drafts, mapping):
    items = {item.annotation_id: item for draft in drafts for item in draft.items}
    rows = []
    for entry in mapping:
        item = items[entry["annotation_id"]]
        for reviewer in (REVIEWER_A, REVIEWER_B):
            rows.append(
                {
                    "review_item_id": entry["review_item_id"],
                    "reviewer_id_hash": reviewer,
                    "judged_axis": item.axis.value,
                    "judged_relation": item.semantic_change.expected_relation.value,
                    "fluent": True,
                    "single_axis": True,
                    "accept": True,
                    "model_output_seen": False,
                }
            )
    return rows


def test_review_packet_hides_design_labels_and_author(tmp_path) -> None:
    _, packet_path, _, _ = _packet(tmp_path)
    rows = [
        json.loads(line)
        for line in packet_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 3
    assert set(rows[0]) == {
        "schema_version",
        "review_item_id",
        "language",
        "context_text",
        "base_text",
        "transformed_text",
    }


def test_two_blind_reviews_compile_to_human_verified_orbits(tmp_path) -> None:
    drafts, _, mapping_path, mapping = _packet(tmp_path)
    orbits = compile_reviewed_orbits(
        drafts,
        mapping,
        _reviews(drafts, mapping),
        review_protocol="blind-frame-v1",
        mapping_sha256="d" * 64,
    )

    validate_human_orbits(orbits)
    assert len(orbits) == 1
    assert all(item.verification_status.value == "human" for item in orbits[0].interventions)
    assert mapping_path.is_file()


def test_compiler_rejects_reviewers_who_saw_model_output(tmp_path) -> None:
    drafts, _, _, mapping = _packet(tmp_path)
    reviews = _reviews(drafts, mapping)
    reviews[0]["model_output_seen"] = True

    with pytest.raises(ValueError, match="saw model output"):
        compile_reviewed_orbits(
            drafts,
            mapping,
            reviews,
            review_protocol="blind-frame-v1",
            mapping_sha256="d" * 64,
        )


def test_compile_run_writes_auditable_unlocked_corpus(tmp_path) -> None:
    drafts, _, mapping_path, mapping = _packet(tmp_path)
    drafts_path = tmp_path / "drafts.jsonl"
    reviews = _reviews(drafts, mapping)
    review_a = tmp_path / "review-a.jsonl"
    review_b = tmp_path / "review-b.jsonl"
    _write_jsonl(review_a, reviews[::2])
    _write_jsonl(review_b, reviews[1::2])

    output, manifest = compile_run(
        drafts_path,
        mapping_path,
        [review_a, review_b],
        tmp_path / "reviewed.jsonl",
        collection_protocol=PROTOCOL,
        review_protocol="blind-frame-v1",
        overwrite=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert output.is_file()
    assert payload["human_authored"] is True
    assert payload["locked"] is False
    assert payload["required_reviewers"] == 2
    with pytest.raises(ValueError, match="at least 96 human orbits"):
        intake_run(
            output,
            tmp_path / "locked.jsonl",
            protocol_version=PROTOCOL,
            evaluation_commit="2" * 40,
            overwrite=False,
        )


def test_dry_run_kit_is_balanced_and_cannot_be_loaded_as_evidence(tmp_path) -> None:
    output_dir = tmp_path / "dry-run"
    outputs = make_dry_run_kit(
        output_dir,
        protocol_version="human-frame-v1-dry-run",
        seed=141421,
        overwrite=False,
    )
    manifest = json.loads(
        (output_dir / "dry-run-manifest.json").read_text(encoding="utf-8")
    )
    packet_paths = sorted(output_dir.glob("author-slot-*.drafts.jsonl"))
    rows = [
        json.loads(line)
        for path in packet_paths
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(outputs) == 5
    assert len(packet_paths) == 3
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) == 4
        for path in packet_paths
    )
    assert manifest["case_count"] == 12
    assert manifest["cases_per_axis"] == {
        "direction": 3,
        "modality": 3,
        "relation": 3,
        "scope": 3,
    }
    assert manifest["cases_per_source_group"] == {
        "dialogue": 4,
        "institutional": 4,
        "narrative": 4,
    }
    assert manifest["dry_run_only"] is True
    assert manifest["model_evaluation_forbidden"] is True
    assert manifest["packet_sha256"] == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in packet_paths
    }
    assert all(row["example_only"] is True for row in rows)
    assert all(row["draft_status"] == "incomplete" for row in rows)
    with pytest.raises(ValueError, match="example-only"):
        load_drafts(packet_paths[0], protocol_version="human-frame-v1-dry-run")
