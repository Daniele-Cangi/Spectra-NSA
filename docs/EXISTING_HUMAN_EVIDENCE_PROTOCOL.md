# Existing Human Evidence Protocol v1

Status: **Stage A frozen; final evaluation metrics have not been computed or
inspected.** The immutable Stage A identity is the commit containing this file.
Stage B must be launched from a clean checkout of that exact commit and must pass
its full SHA through `--protocol-commit`.

Protocol ID: `existing-human-evidence-v1`

## Question and evidentiary boundary

The primary question is whether response scalars, response spectra, or explicit
semantic/frame coordinates contain information about human-labelled semantic
compatibility and frozen-encoder errors beyond cosine and elementary text
diagnostics. Absolute benchmark leadership is not the target.

This is public **development evidence**. It is not a substitute for the future
locked human test: all selected datasets are public and a pretrained model may
have encountered their examples, sources, or distribution.

The source-seeded/human-locked v4 lane remains frozen. Its three author bundles,
thresholds, evaluator target, review protocol, and evidence boundary are not
changed. The new CLI rejects paths containing `source-seed`, `source-seeded`,
`human-locked`, `author-bundle`, or `pilot-v4`. No model or evaluator may be run
on those dry-run materials.

No custom Spectra encoder is trained. There is no fine-tuning, student, mixer, or
large learned classifier in this protocol.

## Dataset audit and selection

The committed machine-readable audit is
`protocols/existing-human-evidence-v1.dataset-audit.json`. It records the bytes
and SHA-256 digest of every local raw input, raw record/group counts, label
counts, raw group overlaps, and counts after leakage control. Raw dataset files
remain outside Git.

| Dataset | Frozen source | License | Authority and schema | Role |
|---|---|---|---|---|
| CONDAQA | official repository commit `bd4857f1f2819937f1505925ad8c809544e8e5ef`; `train.json`, `dev.json`, `test.json` hashes in the audit | Apache-2.0 | One JSON object per line. `sentence1` passage, `sentence2` question, human `label`, `PassageID`, `QuestionID`, `PassageEditID`, `SampleID`. Only complete four-member edit sets are eligible. | Genuine human-authored contrast orbits probing negation and scope. |
| PAWS-Wiki labelled-final | official Google-research-datasets Hugging Face mirror snapshot `161ece9501cf0a11f3e48bd356eaa82de46d6a09`; frozen Parquet hashes in the audit | Repository-specific permissive dataset notice, tagged `other` by the mirror; not Apache/MIT | `id`, `sentence1`, `sentence2`, binary human judgement `label`; official train/validation/test. Only `labeled_final` is used. | High-overlap paraphrase/non-paraphrase collisions with human verification. |
| ANLI v1.0 | official archive `anli_v1.0.zip`, repository commit `b5ce27af54a53bd495af7260fdc485000eed15f8`; archive hash in the audit | CC BY-NC 4.0 | JSONL per R1/R2/R3 and split: `uid`, `context`, `hypothesis`, human `label` (`e`, `n`, `c`) plus metadata. | Human/model-adversarial logical and relational inference. |

Authoritative project pages:

- CONDAQA: <https://github.com/AbhilashaRavichander/CondaQA>
- PAWS: <https://github.com/google-research-datasets/paws> and the frozen
  labelled-final mirror at
  <https://huggingface.co/datasets/google-research-datasets/paws>
- ANLI: <https://github.com/facebookresearch/anli>

MoNLI is not part of v1. Its controlled substitutions are useful diagnostics,
but they add less independent primary human evidence than ANLI and overlap the
negation role already covered by CONDAQA. Synthetic corpora remain secondary
diagnostics only.

### Ground truth mapping

- CONDAQA: within each `(PassageID, QuestionID)` complete orbit, pair the
  query-conditioned original passage with each of the three human edits. The
  compatibility target is one exactly when the human answer string is unchanged
  after normalization. The answer is authority but is never a feature.
- PAWS-Wiki: label one is compatible/paraphrase; zero is incompatible.
- ANLI: entailment (`e`) is compatible; neutral (`n`) and contradiction (`c`)
  are incompatible. Round is an evaluation slice, never a feature.

CONDAQA edit roles are used only to assemble the orbit and define reported test
slices. They cannot enter a feature matrix.

## Frozen splits and sampling

Official train/dev/test boundaries are retained. A source family is then defined
as follows: CONDAQA `(PassageID, QuestionID)` orbit; PAWS normalized unordered
sentence-pair SHA-256; ANLI normalized premise SHA-256.

The structural audit found substantial ANLI premise reuse between official
splits. Leakage control therefore gives test priority, removes test families
from dev, and removes both test and remaining-dev families from train. This
produces zero cross-split group overlap. The same generic control runs on all
three datasets.

Within each controlled split, groups are ordered by SHA-256 of
`spectra-existing-human-evidence-v1 + group_id`. Whole groups are admitted up to
the following caps; no group is split:

| Dataset | Train | Dev/calibration | Test |
|---|---:|---:|---:|
| CONDAQA | 3,000 | 600 (588 available) | 1,200 |
| PAWS-Wiki | 3,000 | 1,000 | 1,200 |
| ANLI | 3,000 | 1,000 | 1,200 |

No test-dependent resampling or class balancing is allowed. Logistic class
weights compensate only inside the training objective.

## Frozen models

| Role | Model and immutable revision | Encoding rule |
|---|---|---|
| Encoder A | `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Raw embeddings retained for norms; L2 normalization only for cosine/response geometry. |
| Encoder B | `intfloat/e5-base-v2@f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | Symmetric-pair rule: prefix every encoded string with `query: `; raw embeddings retained as above. |
| NLI baseline/teacher | `cross-encoder/nli-deberta-v3-small@fa2804872c3b4bd748f38c0185cc85775361e735` | Directional entailment for ANLI; minimum of both directional entailment probabilities for equivalence tasks. |

There is no fine-tuning. MiniLM explicitly reports training on SNLI/MNLI and a
large mixture of public pairs; E5 and the NLI model also inherit broad public
corpus familiarity. This prevents claims of pristine generalization.

## Frozen feature ladder

All feature names are audited. Names containing label, answer, target, dataset,
axis, intervention role, generator, template, expected relation, or model label
fail closed.

### F0 — base

Base cosine, raw left/right embedding norms and norm difference, character and
word-token lengths, absolute differences, and length ratios.

### F1 — cheap scalar diagnostics

Pair response norm between normalized embeddings; token-set Jaccard,
containment, vocabulary ratio, normalized sequence-edit similarity; negation,
modal, and punctuation differences. Local corpus density is omitted in v1
because it depends on arbitrary sample composition and is not reproducibly
defined across the three tasks.

### F2 — response spectrum

Only CONDAQA has a valid orbit. The three edited query-conditioned passage
embeddings are projected as tangent responses around the original. The features
are the three singular values and energy ratios, total energy, effective rank,
spectral entropy, top-mode concentration, numerical rank, and the orbit-level
mean/max/variance of response norms. The calculation is label-blind. Orbit role
is not a column. The same orbit spectrum is attached to its three pair outcomes;
individual response norm remains in F1.

### F3 — semantic/frame coordinates

The existing deterministic query-conditioned frame extractor supplies predicate,
unordered-argument, directed-argument, polarity/scope, and modality alignment;
extractor reliability and explicit missingness; a fixed frame distance; and
canonical-span similarity. Slot/span similarity uses the current frozen encoder,
not dataset annotations. For PAWS and ANLI the left text is the query/reference;
for CONDAQA the human question conditions original and edited passages.

### F4 — combined cheap state

The interpretable fixed score is a monotone maximum over base incompatibility,
frame distance, and span incompatibility plus a missing/reliability penalty. It
is evaluated separately. The learned analysis version is logistic regression on
`F0+F1+F3`, and on `F0+F1+F2+F3` only where F2 exists.

### F5 and F6

F5 is frozen NLI on every eligible dev/test pair. F6 routes to NLI when the
learned cheap probability is closest to 0.5. The ambiguity threshold is the dev
quantile corresponding to a maximum 40% route budget; NLI scores do not select
the threshold. Comparators route the same nominal count by deterministic random
hash and by highest base similarity.

## Analysis and tasks

Every learned probe is `StandardScaler` plus class-balanced L2 logistic
regression (`C=1`, LBFGS, seed 1701, maximum 2,000 iterations). The scaler and
coefficients fit on train only. A decision threshold maximizing F1 fits on dev
only. No nonlinear or large classifier is allowed.

Task 1 predicts compatibility. Reports include the full test set, a high-cosine
slice at or above the dev 75th percentile, four cosine bins fixed by dev
quartiles, CONDAQA edit slices, and ANLI round slices when both classes and at
least 20 examples remain.

Task 2 first freezes each encoder's cosine decision threshold on dev. Its binary
error becomes a new target. F0, F0+F1, and F0+F1+F3 logistic probes test whether
the additional coordinates predict error conditional on the cheap base state.

The nested learned comparisons are F0; F0+F1; F0+F1+F3; and, for CONDAQA,
F0+F1+F2 and F0+F1+F2+F3. F2 is never fabricated for PAWS or ANLI.

Because F0+F1+F3 has the same schema across tasks, a leave-one-dataset-out
transfer probe is mandatory: fit and calibrate on the pooled train/dev records
of the other two datasets, then evaluate unchanged on the held dataset test.
This is a hostile generality check, not part of the primary continuation gate.

## Metrics and uncertainty

Primary information metrics are AUROC and AUPRC. Accuracy and F1 use only the
dev-frozen threshold. Also report 10-bin expected calibration error, area under
the risk-coverage curve, failure recall at 5/10/20/40% review budgets, per-slice
results, NLI call fraction, elapsed wall time, logical/model call counts,
throughput, and peak allocated CUDA memory when available.

Incremental comparisons report delta against the immediately simpler probe.
The primary variable delta is F0+F1+F3 minus F0+F1. The spectrum-beyond-variable
delta is F0+F1+F2+F3 minus F0+F1+F3. Confidence intervals use 1,000 deterministic
percentile bootstrap replicates sampled by source group, never by individual
row. A CI is 95% (`2.5%`, `97.5%`).

## Predeclared gates

The semantic-variable phenomenon is provisionally supported only if all hold:

1. F3 improves AUROC or AUPRC by at least 0.03 over F0/F1 on at least two
   genuinely human evaluation datasets/slices.
2. The improvement direction repeats for both frozen encoders.
3. At least one primary grouped-bootstrap delta CI excludes zero.
4. A meaningful improvement remains in a high-similarity slice.
5. The gain remains after the length, token, edit, and lexical diagnostics in
   F0/F1; no forbidden identity/annotation feature is present.

The spectral claim remains central only if F2 adds at least 0.02 AUROC or AUPRC
beyond F1+F3 on at least two relevant human CONDAQA slices (all, paraphrase,
scope, affirmative), or yields a grouped-CI-supported improvement in
risk-coverage/failure recall where AUROC is saturated. Otherwise spectrum is
demoted even if F3 works.

The selective-cascade gate is at least 95% of NLI's AUROC, no more than 40% of
eligible NLI logical calls, and a better AUROC/risk-coverage or cost-quality
tradeoff than both frozen routing comparators. NLI need not be beaten.

## Interpretation policy

- No F2/F3 increment: stop architecture scaling and rethink the hypothesis.
- F3 works but F2 does not: retain semantic factorization and demote spectrum.
- F2 works but F3 does not: demand stronger confound and transfer controls.
- Signal on only one dataset: call it task-specific monitoring, not general
  semantic reliability.
- Cheap features predict failure but NLI dominates: continue only the routing
  hypothesis.
- NLI is required almost everywhere: do not build a student.

Any implementation defect discovered after this freeze must be documented and
fixed in a new versioned Stage A commit before rerunning. The present protocol
must not be silently edited in response to Stage B metrics.

## Execution and persisted evidence

Structural audit (safe before Stage A):

```powershell
spectra-existing-human-evidence audit `
  --raw-root <external-raw-root> `
  --output protocols/existing-human-evidence-v1.dataset-audit.json
```

Stage B, from the clean Stage A commit:

```powershell
spectra-existing-human-evidence run `
  --raw-root <external-raw-root> `
  --output results/existing-human-evidence-v1 `
  --protocol-commit <full-stage-a-sha> `
  --device cuda
```

The runner refuses dirty/floating checkouts and existing output. It persists
configuration, environment/package/model revisions, dataset hashes, sample
counts, timing/calls/memory, metrics/CIs, and per-example labels, features,
scores, NLI score, cascade decision, and group/slice metadata. The local SQLite
embedding caches are reproducible intermediates and are excluded from the
committed result manifest.
