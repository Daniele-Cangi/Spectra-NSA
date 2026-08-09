# Pilot v4 source-seeded collection protocol

## Status and boundary

This protocol supersedes the manual-source dry run in
[`PILOT_V4_HUMAN_COLLECTION_PROTOCOL.md`](PILOT_V4_HUMAN_COLLECTION_PROTOCOL.md)
before any author packet is distributed. It does not change the frozen observer,
encoder, NLI fallback, thresholds, synthetic results, or evaluation target.

Protocol identifiers are:

- rehearsal collection: `human-frame-v2-source-dry-run`;
- future locked collection: `human-frame-v2-source-seeded`;
- blind review: `blind-frame-v1`;
- frozen evaluation target: `254b225cad05ef38a38275c62296f383c510df1b`.

The rehearsal and all machine proposals are development-only and
`claim_eligible: false`. Spectra, MiniLM, the frame observer, and NLI must not be
run on source records, candidates, author drafts, review packets, rejected
items, or the rehearsal corpus.

## What changed

Authors no longer invent the query and base document from a blank page. The
coordinator mines ordinary public prose and inference proposes a minimal span,
query, frame, and primary semantic axis. In the default `seed-only` mode, the
author receives only natural context, the query, the proposed axis, and a
role-specific authoring contract. The author independently writes the
critical, control, and invariant transformations.

Inference output is a discovery proposal, not ground truth. Human authoring,
blind reconstruction of axis and relation, and unanimous review remain the
only route into a future locked collection.

## Source adapter and licensing

The implemented adapter uses the official Stack Exchange API v2.3
`/questions` endpoint with the documented `withbody` filter. It does not scrape
HTML pages. It respects API `backoff`, records quota metadata, and supports
site and date bounds. Relevant official references are the
[questions endpoint](https://api.stackexchange.com/docs/questions),
[question type](https://api.stackexchange.com/docs/types/question),
[filters](https://api.stackexchange.com/docs/filters), and
[throttling rules](https://api.stackexchange.com/docs/throttle).

Every persisted source record contains:

- platform, site, item identifier, canonical HTTPS URL, and retrieval time;
- creation time and the API-provided `content_license`;
- attribution display name and profile URL;
- normalized title and visible question text;
- SHA-256 of the normalized text, tags, and redaction flags.

The coordinator must preserve the API-provided license and attribution. Stack
Exchange's [content licensing policy](https://stackoverflow.com/help/licensing)
explains the license history; the record-level API value remains authoritative
for each collected item. Missing licenses fail deterministic source filtering.

Code, preformatted blocks, and block quotations are dropped. Email addresses,
phone-like strings, and bare URLs are redacted before persistence, including in
titles. Records requiring those redactions are excluded from the current pilot
rather than silently presented as untouched natural prose. Numeric user and
account identifiers are not retained.

## Inference contract

Inference providers implement a small adapter boundary and return strict
structured records containing:

- source identifier, minimal span, surrounding context, and query;
- main proposition and normalized frame fields;
- candidate axes, exactly one primary axis, confidence, ambiguity flags, and
  optional rejection reason;
- predicate family, syntactic form, source domain, operating mode, and
  assistance level;
- transformations and verifier agreement only in `auto-proposal` mode.

Precomputed JSONL is a deterministic replay format. Its manifest identifies the
producer, proposal-file hash, and the fact that producer-side sampling settings
are unavailable when that is the case. The included deterministic heuristic is
a diagnostic baseline, not a production labeler.

## Deterministic validation and selection

Source validation checks provenance, HTTPS URL, license, hash, length, prose
content, redactions, and obvious signature noise. Candidate validation checks
mode, source binding, confidence, query form, span containment, usable control
context, leakage terms, placeholders, and blocking ambiguity. Auto-proposals
add structural checks for three distinct non-placeholder transformations.

Exact source hashes and near-duplicate contexts are deduplicated. Retained
candidates are selected per axis with an explicit diversity objective over
site, source domain, predicate family, syntactic form, context length, and
lexical novelty. Confidence is only a small component. The requested retained
count must exactly equal the four-axis target; truncating an imbalanced result
is forbidden.

The seed-only pack additionally requires at least three sites, at least three
source domains, and complete four-axis coverage inside every source domain.

## Operating modes

### Seed-only — collection path

`seed-only` never emits machine-written transformations. Public author packets
contain natural context and query, proposed axis, authoring contract, and
placeholders for the human triplet. They intentionally omit canonical URL,
attribution, inference confidence, normalized frame, source identifier, and
proposal diagnostics.

The separate private provenance file is coordinator-only. It is needed for
license compliance, audit, and reproducibility, but must not reach blind
reviewers. Author packets are incomplete, `example_only: true`, and rejected by
the production loader until every placeholder is replaced and the author marks
the case complete.

### Auto-proposal — exploration only

`auto-proposal` may emit complete machine triplets, but every row is marked
`machine_generated: true`, `development_only: true`,
`claim_eligible: false`, and `model_evaluation_forbidden: true`. No author or
review packets are created. The human collection loader rejects these rows by
construction.

## Rehearsal commands

Fetch a bounded multi-site source pool:

```bash
spectra-phase0-source-mine fetch \
  --site workplace --site travel --site history --site law \
  --from-date 2024-01-01 --to-date 2026-07-31 --max-items 80 \
  --output source-mining/sources.jsonl \
  --manifest source-mining/sources.manifest.json
```

Mine a deterministic inference replay and retain three cases per axis:

```bash
spectra-phase0-source-mine mine \
  --sources source-mining/sources.jsonl \
  --inference-provider jsonl \
  --inference-proposals source-mining/inference-proposals.seed-only.jsonl \
  --inference-identity "recorded producer identity" \
  --mode seed-only --max-retained 12 --axis-target 3 --seed 173205 \
  --output source-mining/selected.seed-only.jsonl \
  --manifest source-mining/mining-report.json
```

Create three author packets and private provenance:

```bash
spectra-phase0-source-mine pack \
  --candidates source-mining/selected.seed-only.jsonl \
  --mining-manifest source-mining/mining-report.json \
  --output-dir source-seed-kit --mode seed-only \
  --protocol-version human-frame-v2-source-dry-run \
  --author-slots 3 --axis-target 3 --seed 223607
```

Convert the validated kit into three isolated handoff bundles:

```bash
spectra-phase0-source-mine prepare-distribution \
  --kit-dir source-seed-kit \
  --output-dir source-seed-distribution \
  --protocol-version human-frame-v2-source-dry-run
```

This stage verifies every packet against the source-kit manifest, assigns one
cryptographically random pseudonymous author hash per slot, requires exactly
one case per axis in every bundle, and scans the public rows for private
provenance keys. Each ZIP contains only `source-seeds.jsonl` and
`INSTRUCTIONS.md`. The separate `_coordinator-private-assignments.json` must be
completed with the real participant mapping locally and must never be sent to
authors or reviewers. Send exactly one ZIP to each independent author.

Before distribution, run the existing `draft-status` command and require the
expected twelve incomplete example cases, four per packet and three per axis.
After author return, require `ready_for_assembly: true`, then use the unchanged
assembly, blind-review, compilation, and intake stages from the human-locked
protocol with the new collection identifier.

## Natural-pair exploration

Correction and contrast cues in question text are reported only as exploration
hints. They are neither paired revisions nor labels. A future natural-pair
adapter must explicitly retrieve linked answers, comments, or revision history,
preserve their separate provenance, and place every proposed pair behind human
review. No cue-derived pair enters this rehearsal.

## Freeze rule

The twelve-case rehearsal is discarded after workflow validation. A future
locked run still requires 96 accepted human-authored orbits, three independent
authors, three source groups, all-axis coverage for every author and source
group, and two independent unanimous blind reviews per intervention. Only the
final frozen manifest can authorize the one-shot evaluation against the
unchanged target commit.
