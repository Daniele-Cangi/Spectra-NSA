# Phase 0 pilot v3: semantic variables and fixed aggregation

## Status

This is a development-only falsification study. It tests whether task-relative semantic variables
are a better measurement object than a single embedding displacement or an unconstrained fusion of
response features. It is not claim-eligible evidence of natural-language generalization.

The central result is positive but narrow: a fixed norm over explicit, query-conditioned variables
transfers better than the previous learned fusion. The current lexical observer is deliberately
simple and does not solve semantic canonicalization.

## Experimental design

The corpus contains:

- 256 orbits;
- four independently worded templates: `parallel`, `leading-event`, `records`, and `telegraphic`;
- 2,816 interventions;
- five controlled axes: entity, polarity, quantity, relation, and time;
- a task-relevant critical edit and a matched irrelevant edit for every axis;
- an additional task-relevant relation paraphrase that must remain invariant;
- zero critical-control token-edit mismatches under the frozen MiniLM tokenizer.

Every intervention carries an oracle `semantic_change` annotation with its axis, frame, before and
after values, value-change flag, and query relevance. The evaluator uses this annotation only for
contract validation and labels. Axis, family, relevance, template, and expected relation are
explicitly excluded from model features.

## State views

### Cheap neural response

The previous document response norm, pooled query similarity response, token-local MaxSim response,
and token edit diagnostics.

### Relational state

The frozen NLI distribution is converted into five interpretable variables:

- support: entailment minus contradiction;
- contradiction probability;
- neutrality probability;
- commitment: entailment plus contradiction;
- normalized entropy.

This is a teacher view, not a cheap Spectra representation.

### Lexical variable state

A family-blind observer selects the document sentence with maximum query-content coverage and
measures:

- content coverage;
- entity coverage;
- normalized numeric coverage;
- polarity alignment;
- scope overlap.

Its routed score is the fixed L-infinity distance between the base and transformed state. It does
not receive the intervention family and has no learned per-axis weights.

## Main results

| View | Grouped AUROC | Matched-pair accuracy | Held-out polarity AUROC | Held-out relation AUROC |
| --- | ---: | ---: | ---: | ---: |
| Cheap neural response | 0.8383 | 0.8753 | 0.4157 | 0.5561 |
| Lexical variable distance | 0.9833 | 1.0000 | 1.0000 | 0.7500 |
| Full lexical state | 0.9833 | 1.0000 | 1.0000 | 0.7500 |
| Selected-span semantic state | 0.9174 | 0.9766 | 1.0000 | 0.9048 |
| Fixed lexical + span distance | 0.9925 | 1.0000 | 1.0000 | 0.8955 |
| NLI relational state | 0.9924 | 1.0000 | 1.0000 | 0.8487 |
| Cheap plus lexical | 1.0000 | 1.0000 | 0.8718 | 0.7357 |
| All views | 1.0000 | 1.0000 | 0.9693 | 0.8853 |

The fixed lexical distance obtains AUROC `0.9833` on every leave-one-template-out split and pairwise
accuracy `1.0`. The unconstrained all-view fusion looks perfect on the grouped split but falls to
AUROC `0.7194` and pairwise accuracy `0.7938` when the `leading-event` template is held out.

This reverses the intuitive complexity ordering: the smallest, fixed aggregation is the most stable
under this template shift.

### Relation canonicalizer follow-up

A follow-up observer first selects the query-relevant sentence and compares only that span with the
query using frozen MiniLM pooled cosine. This raises held-out relation AUROC from `0.75` to `0.9048`
without calling the NLI cross-encoder.

The provisional fixed canonical distance is:

\[
D_{canonical}=D_{lexical}+0.5\max(D_{span},0).
\]

The factor `0.5` maps the maximum cosine drop range of two into the unit range used by lexical
coordinates; it was not fitted per axis. This fixed score reaches:

- grouped AUROC `0.9925`;
- matched-pair accuracy `1.0`;
- held-out polarity AUROC `1.0`;
- held-out relation AUROC `0.8955`;
- leave-one-template-out AUROC `1.0`, `1.0`, `0.9885`, and `1.0`.

The learned lexical-plus-span state obtains a higher grouped AUROC (`0.9991`) but essentially the
same held-out relation result. The fixed score is preferred because it preserves monotonicity and
does not learn axis-specific shortcuts.

## What the variables actually encode

The response Jacobian provides a second diagnostic:

| View | Mean effective rank | Mean absolute cross-axis cosine | Axis-decoding accuracy |
| --- | ---: | ---: | ---: |
| Cheap neural response | 2.27 | 0.59 | 0.987 |
| Lexical variable state | 3.30 | 0.57 | 0.800 |
| NLI relational state | 2.03 | 0.83 | 0.553 |
| Cheap plus lexical | 3.56 | 0.50 | 1.000 |
| All views | 3.77 | 0.60 | 1.000 |

The NLI teacher is an excellent relevance-change detector but a poor typed coordinate system: its
axis responses are strongly aligned. Conversely, cheap neural features decode the axis extremely
well, but much of that separability can come from edit-specific magnitude and wording. Axis
decodability alone is therefore not evidence of causal semantic variables.

The fixed scalar distance has rank one and cannot explain which variable changed. That is expected:
it is the routing norm, not the explanatory state itself.

## The relation limit

The lexical observer distinguishes a relevant relation change from the average of its two controls,
so matched-pair accuracy is `1.0`. Individual lexical-only relation AUROC is only `0.75`, however,
because an exact lexical observer moves for both:

- a true relation change such as `approved -> rejected`;
- a meaning-preserving relation paraphrase such as `authorized -> approved`.

This is the correct failure. The variable formulation is useful, but each coordinate needs semantic
canonicalization. The selected-span observer reduces this failure to held-out AUROC `0.8955`, but
does not eliminate it. Surface token replacement and pooled sentence similarity are useful
measurements, not yet a complete relation state.

## Revised architecture

The next representation should contain independently normalized coordinate observers:

1. entity identity and coreference;
2. numeric value, unit, and comparator algebra;
3. normalized temporal interval and ordering;
4. polarity, modality, and scope logic;
5. canonicalized predicate and argument structure;
6. per-coordinate extraction quality or missingness.

Routing should begin with a fixed monotone norm over normalized coordinate mismatches. A learned
fusion may adjust calibration, but it must not be allowed to reverse or hide a coordinate without a
held-generator justification.

In compact form:

\[
D(q,d)=\max_j \left(r_j\,\delta_j(q,d)\right),
\]

where `delta_j` is a normalized mismatch for semantic variable `j` and `r_j` is the reliability of
that coordinate's extractor. The cross-encoder is called when `D` or an extractor's missingness is
ambiguous, not for every pair.

## Decision

Continue the variable formulation and the fixed canonical distance. Do not continue the
unconstrained feature-concatenation path. The next scientific bottleneck is completing relation
canonicalization across paraphrases, not inventing a larger response head.

The next locked pilot must add aliases, coreference, units, numeric equivalence, temporal ranges,
negation scope, modality, argument reversal, and human-authored paraphrases. A successful model must
retain the fixed-norm transfer behavior while reducing the relation-paraphrase false positives.
