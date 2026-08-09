# Pilot v4 source-mining findings

## Decision

Proceed with source-grounded, inference-assisted seed discovery, but keep the
human triplet and blind review as the evidentiary boundary. The real pilot shows
that natural public text can supply a diverse twelve-case rehearsal. It does
not show that fully automatic labeling or transformation is reliable.

No Spectra, MiniLM, frame-observer, or NLI evaluation was run on any material in
this study.

## Source run

On 2026-08-09 the official Stack Exchange API adapter fetched 80 questions: 20
each from Workplace, Travel, History, and Law, bounded from 2024-01-01 through
2026-07-31. The persisted source file has SHA-256
`dfaa19b3e7618d39d18f2d52ed853360d38d513330a14a9980da3c7cb6a0cf52`.

Deterministic filtering retained 69 of 80 records (`86.25%`):

| Rejection | Count |
| --- | ---: |
| Email, phone, or URL redacted | 7 |
| Fragment or non-prose | 2 |
| Source too long | 1 |
| Missing record-level license | 1 |

This conservative policy trades recall for a cleaner and auditable rehearsal.
It should be revisited only by changing the protocol, not by silently relaxing
validation for selected examples.

## Three assistance levels

### A — source filtering only

Level A successfully turns 80 public question records into 69 usable natural
texts with provenance and record-level licenses. It supplies no semantic axis,
query, or intervention.

### B — query, frame, and axis proposal

The deterministic heuristic proposed one candidate for every usable source,
but all 69 were below the predeclared `0.60` confidence threshold. Its proposed
axis distribution was strongly skewed:

| Axis | Proposed |
| --- | ---: |
| Direction | 37 |
| Relation | 17 |
| Scope | 9 |
| Modality | 6 |

This is evidence against treating a keyword heuristic as an automatic labeler.

A human-inspected structured inference replay contained 15 proposals over 13
of the 69 usable sources (`18.84%` source coverage). All 15 passed deterministic
candidate validation; near-duplicate removal left 13, and diversity selection
retained 12. Two proposals were duplicates (`13.33%` of valid proposals), and
three carried non-blocking ambiguity flags (`20%`).

The retained kit has exactly three cases per axis. It spans four sites and is
balanced across three domains:

| Dimension | Distribution |
| --- | --- |
| Axis | relation 3, direction 3, scope 3, modality 3 |
| Site | Workplace 4, Travel 4, History 2, Law 2 |
| Domain | civic 4, professional 4, travel 4 |

Every domain covers all four axes. Each of three author packets contains four
cases, one per axis. The initial status is intentionally fail-closed: 12
incomplete example cases and 120 placeholder values, with
`ready_for_assembly: false`.

The replay is a curated pilot sample, not an estimate of end-to-end inference
recall. Producer-side sampling was host-managed and is unavailable to replay;
the manifest records that limitation and hashes the proposal file.

### C — full triplet auto-proposal

The separate development-only C run contains four source-grounded proposals,
one per axis. All four pass deterministic structure checks. A separate
same-session inspection agreed with three of four (`75%`); the direction case
failed agreement. The run covers three sites and three domains, but inference
was attempted on only four of 69 usable sources (`5.80%`).

This is not an independent verifier study. It is a small feasibility probe, and
the direction disagreement is the more informative observation. All C outputs
are machine-marked, development-only, excluded from human intake, and forbidden
from model evaluation.

## Diversity and natural-pair observations

The final B selection includes service provision, recording preference,
territorial transfer, role attribution, rerouting, work assignment, numeric
scope, exception scope, social-custom scope, physical possibility, request
modality, and travel possibility. The selectors therefore escaped the narrow
synthetic grammar without choosing solely by confidence.

A coarse correction/contrast cue scan found 52 sentences in the raw questions.
These are only hints. Question text alone does not establish a reply pair,
revision pair, or semantic label. The next lateral experiment should extend the
adapter to linked answers, comments, and revision histories, then test whether
human reviewers can recover naturally occurring change/preservation pairs.

## Consequence

The source-seeded workflow is worth continuing because it removes blank-page
authoring and injects natural syntax while preserving a human evidentiary lock.
The next valid action is distribution of the twelve seed-only rehearsal packets
to three independent human authors. Automatic triplets should remain a separate
research branch until a genuinely independent verifier and a much larger,
predeclared sample show acceptable axis-specific reliability.
