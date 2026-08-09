# Phase 0 synthetic pilot: 256-orbit findings

Date: 2026-08-09

Status: development-only diagnostic. This experiment is not claim-eligible.

## Purpose

This pilot tests the measurement pipeline and attempts to falsify easy versions of the semantic
response hypothesis before collecting a larger or human-verified dataset. It uses one factorial
synthetic template, so generator wording, field position, and intervention identity are not diverse.

## Frozen run identity

- orbits: 256;
- interventions: 3,072, twelve per orbit;
- unique encoded texts: 3,326 out of 3,328 requests;
- encoder: `sentence-transformers/all-MiniLM-L6-v2`;
- encoder revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- PyTorch: `2.13.0+cu130`;
- Sentence Transformers: `5.7.0`;
- GPU: NVIDIA GeForce RTX 2060;
- primary weighting: uniform;
- retained spectrum rank: 8;
- numerical spectral floor: `atol=1e-6`, `rtol=1e-5`;
- fresh batched runtime: approximately 40 seconds.

The input, features, summaries, cache statistics, runtime fingerprint, and hashes are recorded in the
generated manifests under the pilot work directory.

## Matched intervention diagnostics

The table compares the unweighted tangent response norm for the relation-changing intervention and
its preserving control within the same family. Token edit is normalized by the longer token sequence.

| Family | Critical mean | Control mean | Critical > control | Critical token edit | Control token edit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Entity | 0.7481 | 0.1527 | 100.0% | 0.0595 | 0.0646 |
| Negation | 0.2735 | 0.0530 | 100.0% | 0.0676 | 0.0944 |
| Quantity | 0.0698 | 0.1176 | 0.0% | 0.0300 | 0.0708 |
| Time/modality | 0.0608 | 0.0333 | 93.75% | 0.0281 | 0.0676 |

Surface interventions have zero token edit and zero response for this uncased encoder. This is a
tokenizer property, not evidence that arbitrary surface noise will be harmless.

## Weighting ablation

| Orbit statistic, mean | Uniform | Strength-weighted | Difference |
| --- | ---: | ---: | ---: |
| Total response energy | 0.7328 | 0.6070 | -0.1258 |
| Effective response rank | 2.4153 | 1.8311 | -0.5842 |
| Critical/invariant bounded contrast | 0.7639 | 0.9403 | +0.1764 |
| Critical/invariant log energy ratio | 2.0387 | 3.5133 | +1.4746 |

Strength weighting substantially inflates the apparent critical-versus-invariant separation because
the pilot strengths are correlated with the expected relation. Uniform weighting is therefore the
primary setting. Strength weighting remains only as an explicitly labelled ablation.

## What failed

The quantity family reverses the expected ordering in every orbit. The relevant change
`1,000 -> 1,500` substitutes one numeric token, while the nominally preserving control
`1,000 -> 1000` changes the tokenizer segmentation from three numeric tokens to one. The control has
more than twice the normalized token edit and produces the larger embedding response.

This falsifies the naive claim that perturbation magnitude alone tracks semantic importance. It also
shows that character-level edit matching is insufficient: both quantity changes have character edit
distance one.

## What remains interesting

Entity and negation critical changes produce larger responses in all 256 matched pairs even though
their token edits are not larger than their controls. Time/modality shows the same direction in
93.75% of pairs. These results justify a better pilot, but they do not establish generalization:
field position and generator identity are constant, and the sentences come from one template.

Global response norm and normalized token edit have Pearson correlation about 0.291 across all
interventions. Within quantity changes the correlation is about 0.718, and within time changes about
0.805. Tokenization must therefore be treated as a primary confounder.

## Decision

Do not scale this generator unchanged to 2,000 orbits. Build pilot v2 with token-matched relevant and
irrelevant fields:

1. use one-token organization and reporter aliases for matched entity substitutions;
2. pair a relevant quantity with an irrelevant report identifier whose replacement has the same
   token edit pattern;
3. pair the operational year with an irrelevant publication year;
4. diversify templates and field positions;
5. freeze generator families before examining the next locked summary;
6. repeat on `intfloat/e5-base-v2` only after the primary encoder passes the matched-control test.

The continuation target for pilot v2 is directionally correct critical-versus-control response in at
least three critical families after token-edit matching, without strength weighting. Collision AUROC
and distillation remain out of scope until that condition is met.
