# Phase 0 pilot v2: task-relative response fibers

## Status

This is a development-only falsification study. It uses 256 synthetic orbits generated from one
factorial template family. The results guide the next experiment, but they are not claim-eligible
evidence of generalization or novelty.

The pilot changed the working hypothesis. A single document-level response spectrum is not a
general semantic diagnostic. Different failure families require different, query-conditioned views,
and a frozen NLI cross-encoder is already a much stronger teacher on this synthetic task.

## Why a second pilot was necessary

The first 256-orbit pilot exposed a tokenization confound: some critical and preserving numerical
edits did not have matched token edit distances. V2 replaces those examples with parallel facts:

- one shipment fact is relevant to the query;
- one report-metadata fact is irrelevant to the query;
- a critical edit changes the relevant fact;
- its matched control changes the irrelevant fact using the same intervention type;
- each orbit carries the relevant shipment clause as `context_text`.

Entity, quantity, and time pairs have exactly the same normalized token edit distance
(`0.032258`); negation pairs are also exactly matched within their family (`0.03125`). Each orbit
contains 12 interventions and the corpus contains 3,072 interventions in total.

## Frozen models

- sentence encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- NLI baseline: `cross-encoder/nli-deberta-v3-small`, revision
  `fa2804872c3b4bd748f38c0185cc85775361e735`;
- runtime: PyTorch `2.13.0+cu130`, Sentence Transformers `5.7.0`, CUDA on an RTX 2060.

All weights are frozen. No result below comes from fine-tuning either model.

## Views evaluated

### Document-only tangent response

The original measurement: encode the full document, project each intervention response onto the
unit-sphere tangent plane, and compare its norm and spectrum.

### Pooled query-conditioned response

Measure the signed and absolute change in cosine similarity between the query and the pooled
document embedding.

### Token-local late interaction

Use MiniLM token embeddings and score the query against the document with mean query-token MaxSim,
similar to a late-interaction retrieval view. Measure the score change under each intervention.

### Feature fusion

Fit a standardized logistic regression using document response norm, pooled query-response fields,
late-interaction response fields, and normalized token edit distance. Split by orbit, never by
individual intervention.

### Frozen NLI cross-encoder

Score the document as the premise and the query clause as the hypothesis. A critical intervention
should reduce entailment more than its matched irrelevant control.

## Results

The main diagnostic is the fraction of orbits in which the critical edit moves the score more than
its matched preserving control.

| Family | Document response norm | Pooled query response | Token-local response | NLI entailment drop |
| --- | ---: | ---: | ---: | ---: |
| Entity | 39.84% | 61.33% | 97.27% | 100% |
| Negation | 100% | 3.13% | 31.25% | 100% |
| Quantity | 3.13% | 30.86% | 96.48% | 100% |
| Time/modality | 0% | 75.78% | 100% | 100% |

The pooled and token-local columns use absolute score change. Signed changes strengthen the entity,
quantity, and time result for late interaction: the critical edit is more negative in 99.61%,
99.22%, and 100% of orbits respectively. Negation remains unresolved at 26.56%.

The document-only critical-versus-invariant energy contrast has mean `-0.1986`. This directly
falsifies the assumption that a universal scalar response magnitude should increase for every
task-critical change.

## Fusion result and its trap

The grouped orbit split contains 179 training and 77 test orbits. The combined model reaches:

- AUROC `0.99783`;
- average precision `0.99845`;
- accuracy `0.99026`.

That number is misleading without a harder split. Leave-one-family-out AUROC is:

| Held-out family | AUROC |
| --- | ---: |
| Entity | 1.0000 |
| Negation | 0.3539 |
| Quantity | 0.9969 |
| Time/modality | 1.0000 |

The grouped model learns the family-specific negation pattern because examples from that family
occur in both train and test. It does not discover a universal response law that transfers to unseen
negation behavior. The standardized coefficient of raw document response norm is only `-0.0173`,
so that original scalar contributes almost nothing to the combined classifier.

## Strong-baseline result

The frozen NLI cross-encoder separates the critical edit from the matched irrelevant control in all
256 orbits for every family. Mean critical entailment drops are approximately:

- entity: `0.9798`;
- negation: `0.9797`;
- quantity: `0.9707`;
- time/modality: `0.9711`.

This makes the current synthetic pilot too easy for a relation-aware cross-encoder. It also prevents
an accuracy-based novelty claim for the current response features. Spectra must offer a different
advantage: lower cost, shortlist routing, calibrated failure detection, transferable auditing, or a
useful single-pass approximation of the stronger teacher.

## Revised architecture: typed response fibers

The next Spectra representation should not collapse all interventions into one scalar or one shared
response geometry. It should expose a small bundle of task-conditioned response fibers:

1. **global fiber** -- pooled embedding displacement and low-rank tangent spectrum;
2. **local fiber** -- query-token/document-token late-interaction response;
3. **polarity fiber** -- contradiction, entailment, scope, and negation response learned from an NLI
   teacher;
4. **typed value fibers** -- explicit entity, number, unit, comparator, and temporal compatibility;
5. **router** -- a calibrated gate that selects or combines fibers for each query-document pair.

The mathematical object is therefore a typed, query-conditioned response bundle rather than a
universal spectrum:

\[
\mathcal{R}(q,d) =
\{R_{global}(q,d), R_{local}(q,d), R_{polarity}(q,d), R_{value}(q,d)\}.
\]

A future heterogeneous commutator remains viable only as a student representation distilled from
these measured teachers. Its target is no longer an undifferentiated singular-value vector.

## Revised research question

> Can a single-pass typed response student approximate the useful decisions of token-local and NLI
> teachers, detect their failure modes, and route only ambiguous pairs to expensive cross-encoding?

This is more demanding and more useful than asking whether a perturbation spectrum alone predicts
semantic change.

## Next experimental gates

Do not scale the present template count and call it validation. The next experiment must:

1. use multiple independently written generators per intervention family;
2. hold out complete generators and lexical inventories;
3. include human-authored natural retrieval collisions;
4. add numerical, unit, comparator, scope, and argument-reversal cases that NLI may not solve
   uniformly;
5. compare quality, latency, memory, and encoder calls against frozen NLI and late interaction;
6. test whether a compact student preserves collision recall at a fixed cross-encoder review budget;
7. report calibration and risk-coverage, not only AUROC;
8. abandon or narrow the response-spectrum component if it adds no value beyond the teachers and
   cheap scalar controls.

## Decision

Continue the project, but not by defending the original single-spectrum hypothesis. Keep the
measurement harness, the confounder controls, and the response views. Reframe the model around typed
fibers, use NLI as a teacher and upper baseline, and make cost-aware selective routing the primary
product hypothesis.
