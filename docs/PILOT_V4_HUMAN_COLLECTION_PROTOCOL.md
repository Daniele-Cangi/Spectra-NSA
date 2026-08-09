# Pilot v4 human-locked collection protocol

## Frozen status

Protocol identifiers:

- collection: `human-frame-v1`;
- review: `blind-frame-v1`;
- evaluation target: commit `254b225cad05ef38a38275c62296f383c510df1b`;
- encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- NLI fallback: `cross-encoder/nli-deberta-v3-small`, revision
  `fa2804872c3b4bd748f38c0185cc85775361e735`.

The observer, models, thresholds, and evaluation gates are frozen before collection. Collection
tooling added after the evaluation target commit does not modify the observer or its scores.

This protocol creates development-quality human evidence. It remains `claim_eligible: false` until
the collection procedure, consent requirements, and intended public use receive separate approval.

## Target

The final locked set requires at least:

- 96 complete human-authored orbits;
- 1,152 interventions, twelve per orbit;
- three independently identified authors;
- three source groups;
- two independent blind reviews per intervention, at least 2,304 review decisions.

Each orbit must cover relation, direction, scope, and modality. Every axis contains:

- `critical`: a query-relevant semantic change;
- `control`: the same type of change in a query-irrelevant frame;
- `invariant`: a query-relevant rewrite that preserves the task relation.

Authors write the query, base document, and every transformed document themselves. LLM-generated,
machine-paraphrased, copied benchmark, and synthetic-v4-derived text is excluded from the locked set.

## Challenge composition

The set should deliberately leave the synthetic grammar. Across source groups, authors should use:

- active, passive, nominalized, elliptical, and quoted constructions;
- aliases, pronouns, and cross-sentence coreference;
- acquisition, sale, approval, revocation, prevention, causality, and event-state transitions;
- modal auxiliaries, reported belief, counterfactuals, and nested negation;
- distractors sharing entities or predicates with the query;
- natural variation in sentence length and document order.

An author may know the requested axis and role. They must not see model scores, model predictions, or
the synthetic error audit while authoring.

## Pseudonymous identities

The coordinator assigns each participant a random high-entropy identifier and stores the private
identity mapping outside the repository. Dataset records contain only
`sha256:<64 lowercase hex digits>`. Do not hash an email address or name directly: low-entropy
identifiers can be guessed.

An author cannot review any case in the same locked batch. Reviewer identities are checked against
all author hashes, not only the author of the current case.

## Author draft

Drafts are JSONL, one complete orbit per line. The non-runnable schema template is
[`human_frame_draft.template.json`](../examples/human_frame_draft.template.json). Every draft has a
globally unique `case_id`; every intervention has a globally unique `annotation_id`.

The compiler derives the task relation from `query_relevant` and `value_changed`. It rejects a role
whose flags disagree with the fixed contract, duplicate texts, incomplete axes, or incomplete roles.

## Blind review

Create a public review packet and a private mapping:

```bash
spectra-phase0-human-frame-collection review-packet \
  --drafts drafts.jsonl \
  --packet review-packet.jsonl \
  --private-mapping private-review-map.jsonl \
  --protocol-version human-frame-v1 \
  --seed 161803
```

The public packet contains only:

- opaque review-item identifier;
- language;
- query;
- base text;
- transformed text.

It excludes author, annotation identifier, axis, role, expected relation, and all model output. The
private mapping must not be sent to reviewers. Packet order is shuffled and both files are hashed in
a manifest.

Each reviewer independently submits the fields shown in
[`human_frame_review.template.json`](../examples/human_frame_review.template.json). A review is
accepted only when the reviewer:

- did not see model output;
- judges the text fluent;
- judges it to contain one controlled semantic-axis edit;
- independently reconstructs the intended axis and task relation;
- explicitly accepts the item.

Compilation requires unanimous agreement from at least two reviewers. A disagreement blocks the
whole item; labels are never resolved by model score or majority vote after inspecting predictions.

## Compilation and freeze

Compile independently reviewed drafts:

```bash
spectra-phase0-human-frame-collection compile \
  --drafts drafts.jsonl \
  --private-mapping private-review-map.jsonl \
  --reviews reviewer-a.jsonl reviewer-b.jsonl \
  --output reviewed-orbits.jsonl \
  --collection-protocol human-frame-v1 \
  --review-protocol blind-frame-v1
```

The compiled file is human verified but not yet locked. Freeze it only after all target counts are
met:

```bash
spectra-phase0-human-frame-intake \
  --input reviewed-orbits.jsonl \
  --output human-locked-orbits.jsonl \
  --protocol-version human-frame-v1 \
  --evaluation-commit 254b225cad05ef38a38275c62296f383c510df1b
```

The intake command enforces the minimum orbit, author, source-group, and reviewer counts. Its
manifest records the immutable evaluation commit and hashes the complete input and output.

## One-shot evaluation gate

Do not run the frame observer or NLI model on drafts, review packets, rejected items, or the compiled
pre-freeze corpus. Run them once after the locked manifest exists.

The predeclared `human-frame-v1` gate requires:

- overall cascade AUROC at least `0.95`;
- AUROC at least `0.90` on every semantic axis;
- AUROC at least `0.90` for every author and source group;
- critical recall at least `0.95`;
- control false-positive rate at most `0.05`;
- invariant false-positive rate at most `0.10`;
- at least 50% reduction of the selected-span relation error, or relation AUROC at least `0.999`
  when the selected-span baseline is already perfect;
- NLI fallback on at most 30% of interventions.

The evaluator detects the `human-locked` partition and selects this gate automatically.

## Failure policy

If the locked set fails, publish the failure and its untouched error audit. Do not tune thresholds or
the observer against this set. A revised architecture requires a new protocol version and a newly
authored batch; the original locked set remains a terminal test artifact.
