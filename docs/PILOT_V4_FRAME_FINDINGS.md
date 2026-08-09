# Pilot v4: adversarial predicate/argument frames

## Status

This is a development-only synthetic falsification study. It tests whether explicit
predicate/argument coordinates plus selective NLI fallback can repair the relation limit found in
Pilot v3. It is not evidence of unrestricted natural-language generalization.

The result supports a cascade, not a larger learned fusion: deterministic structural coordinates
handle direction, scope, and modality; a frozen NLI teacher is called only when the isolated
predicate coordinate is ambiguous.

## Locked synthetic design

The versioned generator produces:

- 192 orbits and 2,304 interventions;
- eight predicate families;
- four active/passive and leading/reported templates;
- four axes: relation, argument direction, scope, and modality;
- one task-relevant critical edit, one irrelevant matched control, and one task-relevant invariant
  per axis;
- held predicate, held template, and double-holdout partitions;
- a manifest containing the seed, partition counts, and corpus SHA-256.

The observer never receives axis, role, predicate-family, template, partition, or expected relation
as an input. These fields are used only for contract validation and evaluation slices.

This corpus is automatically generated and explicitly marked `human_authored: false` and
`claim_eligible: false`.

## Frame observer

The observer selects the query-relevant sentence and extracts a binary semantic frame:

```text
predicate, actor, patient, polarity, modality, voice, reliability
```

It compares query and candidate using the coordinates:

```text
predicate_alignment
argument_alignment
direction_alignment
scope_alignment
modality_alignment
reliability
```

`argument_alignment` is unordered, while `direction_alignment` preserves actor/patient order. This
is the important separation: `A acquired B` and `B acquired A` contain the same arguments but have
opposite direction.

The fixed distance is a monotone maximum. Predicate cosine drop is divided by two to map its range
to the unit range of the structural coordinates. No per-axis weights are fitted.

## Selective cascade

The frozen thresholds are:

- direct change when frame distance is at least `0.35`;
- direct preservation when frame distance is at most `0.08`;
- NLI fallback inside `(0.08, 0.35)` or when frame reliability is below `0.75`.

On the complete pilot, all 360 fallbacks are relation cases. Direction, scope, and modality require
no NLI calls.

## Main results

| View | Overall AUROC | Accuracy | Critical recall | False-positive rate | Pair accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected span | 0.7975 | 0.7344 | 0.4570 | 0.1270 | 0.9518 |
| Fixed frame | 0.9971 | 0.9271 | 0.7812 | 0.0000 | 1.0000 |
| NLI on every pair | 0.9879 | 0.9301 | 0.8242 | 0.0169 | 0.9948 |
| Frame + selective NLI | 0.9995 | 0.9957 | 0.9961 | 0.0046 | 1.0000 |

Per-axis AUROC:

| Axis | Selected span | Fixed frame | Selective cascade |
| --- | ---: | ---: | ---: |
| Direction | 0.8484 | 1.0000 | 1.0000 |
| Modality | 0.6425 | 1.0000 | 1.0000 |
| Relation | 0.9615 | 0.9531 | 0.9972 |
| Scope | 0.9304 | 1.0000 | 1.0000 |

The cascade uses NLI for 360 of 2,304 interventions (`15.625%`). Its minimum AUROC across the held
predicate, held template, and double-holdout partitions is `0.9995`. The task-relevant invariant
false-positive rate is `0.0091`; irrelevant controls have zero false positives.

NLI on every pair is not the preferred architecture. It is weaker than the structural observer on
modality and adds unnecessary cost to coordinates that already have exact logic.

## Gate correction

The initial draft asked for an absolute relation AUROC improvement of `0.05`. The observed selected
span baseline is `0.9615`, leaving only `0.0385` of mathematical headroom, so that gate cannot be
passed by any model.

The evaluator preserves and reports the failed absolute check. Gate revision v1.1 accepts either:

- an absolute AUROC gain of at least `0.05`; or
- when that gain is mathematically infeasible, a reduction of at least half of the remaining AUROC
  error.

The cascade gains `0.0357` AUROC and removes `92.64%` of the remaining relation error. This correction
was made after observing development metrics, so it cannot upgrade this pilot to claim-eligible
evidence.

## Residual errors

At the frozen threshold there are three false negatives and seven false positives among 2,304
interventions. All are relation fallbacks.

- The false negatives are passive `acquired -> sold` cases. The NLI teacher treats sale as compatible
  or neutral rather than as failure to support acquisition.
- Most false positives are valid relation paraphrases in acquisition, prevention, launch, and
  funding. Here the NLI teacher moves more than the isolated predicate observer.

These cases show why the NLI fallback remains a teacher and not the semantic state itself. The next
relation coordinate should model event transitions and temporal ownership, rather than interpreting
every non-entailment as contradiction.

## Human-data boundary

The repository includes a strict intake command for a future human-authored locked set. It requires
human verification status, provenance fields, unique annotation identifiers, and complete matched
roles for every axis. Freezing a file does not make it claim eligible; an independently approved
collection and evaluation protocol is still required.

The authoring, double-blind-review, freeze, and one-shot evaluation procedure is specified in the
[human-locked collection protocol](PILOT_V4_HUMAN_COLLECTION_PROTOCOL.md).

No generated text in this pilot is represented as human authored.

## Decision

Keep the fixed frame coordinates and selective fallback. Do not train a single-pass student yet.
The synthetic development gate is passed under the feasibility-corrected rule, but the architecture
must next survive a genuinely human-authored locked set with event-state relations, coreference,
aliases, and sentences outside the binary active/passive grammar.
