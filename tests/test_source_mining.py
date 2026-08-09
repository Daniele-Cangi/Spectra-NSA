import hashlib
import json
import re
import zipfile

import pytest

from experiments.human_frame_collection import inspect_draft_packets, load_drafts
from experiments.source_mining import (
    InferenceProposal,
    MinedCandidate,
    SourceRecord,
    StackExchangeAdapter,
    mine_sources,
    pack_candidates,
    prepare_distribution,
    source_rejection_reason,
)


AXES = ("relation", "direction", "scope", "modality")
SITES = ("workplace", "travel", "history")
DOMAINS = ("social", "practical", "factual")


def _source(index: int, *, site: str) -> SourceRecord:
    themes = (
        ("museum board", "loaned a sculpture", "gallery lighting"),
        ("rail operator", "changed a timetable", "station accessibility"),
        ("research council", "funded an archive", "catalogue preservation"),
        ("tenant committee", "approved a repair", "courtyard maintenance"),
        ("ferry company", "cancelled a crossing", "passenger accommodation"),
        ("local historian", "identified a manuscript", "regional chronology"),
        ("school panel", "revised an admissions rule", "library opening hours"),
        ("travel agency", "reserved a mountain lodge", "baggage transfer"),
        ("heritage trust", "restored a chapel", "stonework inspection"),
        ("workshop supervisor", "assigned a safety review", "equipment storage"),
        ("tour coordinator", "extended a walking route", "weather guidance"),
        ("records office", "released a census volume", "index transcription"),
    )
    actor, action, background = themes[index % len(themes)]
    text = (
        f"The {actor} {action} after a public meeting in district {index}. "
        f"A separate participant documented {background} for later discussion. "
        "The account includes enough ordinary background for an independent comparison."
    )
    return SourceRecord(
        source_id=f"stackexchange:{site}:question:{1000 + index}",
        platform="stackexchange",
        site=site,
        item_id=str(1000 + index),
        canonical_url=f"https://{site}.stackexchange.com/questions/{1000 + index}/x",
        retrieved_utc="2026-08-09T00:00:00+00:00",
        created_utc="2025-01-01T00:00:00+00:00",
        content_license="CC BY-SA 4.0",
        source_type="question",
        source_hash=hashlib.sha256(text.encode()).hexdigest(),
        attribution_display_name=f"Contributor {index}",
        attribution_url=f"https://{site}.stackexchange.com/users/{index}/x",
        title=f"What did the {actor} decide in district {index}?",
        text=text,
        tags=("example",),
    )


def _proposal(
    source: SourceRecord,
    *,
    axis: str,
    index: int,
    mode: str = "seed-only",
) -> InferenceProposal:
    span = source.text.split(". ", 1)[0] + "."
    transformations = None
    assistance = "query-frame-axis"
    if mode == "auto-proposal":
        assistance = "full-triplet"
        transformations = {
            role: {
                "transformed_text": f"Development-only {axis} {role} proposal {index}.",
                "frame_id": "distractor" if role == "control" else "target",
                "before_value": "before",
                "after_value": "before" if role == "invariant" else "after",
            }
            for role in ("critical", "control", "invariant")
        }
    return InferenceProposal(
        proposal_id=f"proposal-{axis}-{index}",
        source_id=source.source_id,
        minimal_span=span,
        surrounding_context=source.text,
        query=source.title,
        main_proposition=span,
        actor=f"Project group {index}",
        predicate="approved",
        patient="a distinct proposal",
        polarity="positive",
        modality="asserted",
        scope_cues=(),
        axis_candidates=(axis,),
        primary_axis=axis,
        confidence=0.92,
        ambiguity_flags=(),
        rejection_reason=None,
        predicate_family=f"approval-{index}",
        syntactic_form=("active" if index % 2 else "reported"),
        source_domain=DOMAINS[index % len(DOMAINS)],
        mode=mode,
        assistance_level=assistance,
        transformations=transformations,
        verifier_agreement=(True if mode == "auto-proposal" else None),
    )


class _MockInference:
    def __init__(self, proposals):
        self.proposals = {proposal.source_id: proposal for proposal in proposals}

    @property
    def identity(self):
        return {"provider": "offline-mock", "temperature": 0}

    def infer(self, source):
        proposal = self.proposals.get(source.source_id)
        return (proposal,) if proposal else ()


def _balanced_candidates(mode: str = "seed-only"):
    sources = []
    proposals = []
    index = 0
    for axis in AXES:
        for repetition in range(3):
            source = _source(index, site=SITES[repetition])
            sources.append(source)
            proposals.append(_proposal(source, axis=axis, index=index, mode=mode))
            index += 1
    return sources, proposals


def _nested_keys(value):
    if isinstance(value, dict):
        return set(value).union(*(_nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value))
    return set()


def test_stackexchange_adapter_preserves_license_but_drops_code_and_ids() -> None:
    def transport(url):
        assert "site=workplace" in url
        assert "filter=withbody" in url
        return {
            "items": [
                {
                    "question_id": 42,
                    "link": "https://workplace.stackexchange.com/questions/42/x?utm=x",
                    "title": "Did the manager approve the request?",
                    "body": (
                        "<p>The manager approved the request after the meeting.</p>"
                        "<pre>secret_code()</pre><blockquote>unrelated quote</blockquote>"
                        "<p>A colleague separately documented the schedule and location.</p>"
                        "<p>The situation was described for workplace advice.</p>"
                    ),
                    "creation_date": 1735689600,
                    "content_license": "CC BY-SA 4.0",
                    "tags": ["management"],
                    "owner": {
                        "display_name": "Example Contributor",
                        "link": "https://workplace.stackexchange.com/users/7/x?tab=profile",
                        "user_id": 7,
                        "account_id": 99,
                    },
                }
            ],
            "quota_remaining": 9999,
            "quota_max": 10000,
            "has_more": False,
        }

    records, details = StackExchangeAdapter(
        transport=transport, sleep=lambda _: None
    ).fetch(site="workplace", from_date=None, to_date=None, max_items=1)

    assert len(records) == 1
    record = records[0]
    assert record.content_license == "CC BY-SA 4.0"
    assert record.canonical_url.endswith("/questions/42/x")
    assert "secret_code" not in record.text
    assert "unrelated quote" not in record.text
    assert set(record.to_dict()["attribution"]) == {"display_name", "profile_url"}
    assert "user_id" not in json.dumps(record.to_dict())
    assert details["quota_remaining"] == 9999


def test_source_filter_rejects_missing_record_level_license() -> None:
    source = _source(0, site="workplace")
    source = SourceRecord(**{**source.__dict__, "content_license": "unknown"})

    assert source_rejection_reason(source) == "missing-license"


def test_mining_funnel_deduplicates_and_selects_balanced_diversity() -> None:
    sources, proposals = _balanced_candidates()
    duplicate = SourceRecord(
        **{
            **sources[0].__dict__,
            "source_id": "stackexchange:workplace:question:duplicate",
            "item_id": "duplicate",
            "canonical_url": "https://workplace.stackexchange.com/questions/999/x",
        }
    )
    duplicate_proposal = _proposal(duplicate, axis="relation", index=99)
    sources.append(duplicate)
    proposals.append(duplicate_proposal)

    selected, report = mine_sources(
        sources,
        _MockInference(proposals),
        mode="seed-only",
        maximum=12,
        axis_target=3,
        seed=7,
    )

    assert len(selected) == 12
    assert report["funnel"] == {
        "raw_source_items": 13,
        "usable_text_items": 13,
        "inference_candidates": 13,
        "deterministic_valid_candidates": 13,
        "deduplicated_candidates": 12,
        "balanced_selected_candidates": 12,
    }
    assert report["selected_axis_distribution"] == {
        "direction": 3,
        "modality": 3,
        "relation": 3,
        "scope": 3,
    }
    assert set(report["selected_site_distribution"]) == set(SITES)
    assert report["duplicate_rate"] == pytest.approx(1 / 13)

    with pytest.raises(ValueError, match="balanced axis target"):
        mine_sources(
            sources,
            _MockInference(proposals),
            mode="seed-only",
            maximum=11,
            axis_target=3,
            seed=7,
        )


def test_seed_pack_separates_public_author_material_from_provenance(tmp_path) -> None:
    sources, proposals = _balanced_candidates()
    candidates = [
        MinedCandidate(f"candidate-{index}", source, proposal)
        for index, (source, proposal) in enumerate(zip(sources, proposals))
    ]
    outputs = pack_candidates(
        candidates,
        tmp_path / "kit",
        mode="seed-only",
        protocol_version="human-frame-v2-source-dry-run",
        author_slots=3,
        axis_target=3,
        seed=11,
        inference_identity={"provider": "offline-mock"},
        overwrite=False,
    )
    packet_paths = sorted((tmp_path / "kit").glob("author-slot-*.jsonl"))
    public_rows = [
        json.loads(line)
        for path in packet_paths
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    private_rows = [
        json.loads(line)
        for line in (tmp_path / "kit" / "private-source-provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(outputs) == 6
    assert len(packet_paths) == 3
    assert all(len(path.read_text().splitlines()) == 4 for path in packet_paths)
    assert len(public_rows) == 12
    assert all(row["inference_assisted_seed"] is True for row in public_rows)
    forbidden_public_keys = {
        "canonical_url",
        "attribution",
        "confidence",
        "source_id",
        "inference_proposal",
        "main_proposition",
        "proposal_id",
    }
    assert all(not (_nested_keys(row) & forbidden_public_keys) for row in public_rows)
    assert all("canonical_url" in json.dumps(row) for row in private_rows)
    assert all("attribution" in json.dumps(row) for row in private_rows)
    manifest = json.loads(
        (tmp_path / "kit" / "source-seed-manifest.json").read_text()
    )
    assert manifest["development_only"] is True
    assert manifest["dry_run_only"] is True
    assert manifest["machine_generated"] is False
    report = inspect_draft_packets(
        packet_paths,
        protocol_version="human-frame-v2-source-dry-run",
    )
    assert report["ready_for_assembly"] is False
    assert report["case_count"] == 12
    assert report["placeholder_count"] > 100
    with pytest.raises(ValueError, match="example-only"):
        load_drafts(
            packet_paths[0],
            protocol_version="human-frame-v2-source-dry-run",
        )


def test_auto_proposals_are_machine_marked_and_rejected_by_human_loader(
    tmp_path,
) -> None:
    sources, proposals = _balanced_candidates(mode="auto-proposal")
    candidates = [
        MinedCandidate(f"candidate-{index}", source, proposal)
        for index, (source, proposal) in enumerate(zip(sources, proposals))
    ]
    pack_candidates(
        candidates,
        tmp_path / "auto",
        mode="auto-proposal",
        protocol_version="development-auto-v1",
        author_slots=0,
        axis_target=3,
        seed=13,
        inference_identity={"provider": "offline-mock"},
        overwrite=False,
    )
    path = tmp_path / "auto" / "development-auto-proposals.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert all(row["machine_generated"] is True for row in rows)
    assert all(row["development_only"] is True for row in rows)
    assert all(row["claim_eligible"] is False for row in rows)
    with pytest.raises(ValueError, match="machine-generated"):
        load_drafts(path, protocol_version="development-auto-v1")


def test_distribution_bundles_assign_distinct_authors_without_private_data(
    tmp_path,
) -> None:
    sources, proposals = _balanced_candidates()
    candidates = [
        MinedCandidate(f"candidate-{index}", source, proposal)
        for index, (source, proposal) in enumerate(zip(sources, proposals))
    ]
    kit = tmp_path / "kit"
    pack_candidates(
        candidates,
        kit,
        mode="seed-only",
        protocol_version="human-frame-v2-source-dry-run",
        author_slots=3,
        axis_target=3,
        seed=17,
        inference_identity={"provider": "offline-mock"},
        overwrite=False,
    )

    outputs = prepare_distribution(
        kit,
        tmp_path / "distribution",
        protocol_version="human-frame-v2-source-dry-run",
        overwrite=False,
    )

    assert len(outputs) == 5
    bundles = sorted((tmp_path / "distribution").glob("author-slot-*.zip"))
    assert len(bundles) == 3
    author_hashes = set()
    for bundle in bundles:
        with zipfile.ZipFile(bundle) as archive:
            assert set(archive.namelist()) == {
                "INSTRUCTIONS.md",
                "source-seeds.jsonl",
            }
            assert "coordinator" not in " ".join(archive.namelist()).lower()
            rows = [
                json.loads(line)
                for line in archive.read("source-seeds.jsonl")
                .decode("utf-8")
                .splitlines()
            ]
        assert len(rows) == 4
        assert {row["target_axis"] for row in rows} == set(AXES)
        assert len({row["author_id_hash"] for row in rows}) == 1
        author_hash = rows[0]["author_id_hash"]
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", author_hash)
        author_hashes.add(author_hash)
        assert all("attribution" not in _nested_keys(row) for row in rows)
        assert all("canonical_url" not in _nested_keys(row) for row in rows)
    assert len(author_hashes) == 3

    private = json.loads(
        (tmp_path / "distribution" / "_coordinator-private-assignments.json")
        .read_text()
    )
    assert len(private["assignments"]) == 3
    manifest = json.loads(
        (tmp_path / "distribution" / "distribution-manifest.json").read_text()
    )
    assert manifest["bundle_count"] == 3
    assert manifest["case_count"] == 12
    assert manifest["model_evaluation_forbidden"] is True


def test_distribution_refuses_a_tampered_source_packet(tmp_path) -> None:
    sources, proposals = _balanced_candidates()
    candidates = [
        MinedCandidate(f"candidate-{index}", source, proposal)
        for index, (source, proposal) in enumerate(zip(sources, proposals))
    ]
    kit = tmp_path / "kit"
    pack_candidates(
        candidates,
        kit,
        mode="seed-only",
        protocol_version="human-frame-v2-source-dry-run",
        author_slots=3,
        axis_target=3,
        seed=19,
        inference_identity={"provider": "offline-mock"},
        overwrite=False,
    )
    packet = kit / "author-slot-01.source-seeds.jsonl"
    packet.write_text(packet.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        prepare_distribution(
            kit,
            tmp_path / "distribution",
            protocol_version="human-frame-v2-source-dry-run",
            overwrite=False,
        )
