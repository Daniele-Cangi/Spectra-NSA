# Phase 0: Semantic Response Measurement

## Decision record

Phase 0 tests the proposed measurement object before building Spectra v3. It uses frozen, existing
encoders and asks whether local response spectra contain information that ordinary embedding
diagnostics miss.

This protocol makes the following initial decisions:

- language: English for the first controlled experiment;
- task focus: dense retrieval and semantic-collision detection;
- primary encoder: `sentence-transformers/all-MiniLM-L6-v2` for cheap iteration;
- validation encoder: `intfloat/e5-base-v2` to test whether the phenomenon transfers across model
  scale and training recipe;
- representation: normalized sentence embedding;
- decomposition: tangent-projected response matrix followed by truncated SVD;
- architecture work: prohibited until the measurement gate is evaluated.

These are experimental choices, not permanent product constraints.

## Phase 0 questions

### Q1: Does response geometry exist beyond scalar sensitivity?

Compare the full response spectrum and subspace features with simple response variance and average
embedding displacement.

### Q2: Does it predict failures?

Test whether the response features predict false high-similarity pairs and retrieval errors after
conditioning on base cosine similarity.

### Q3: Does it transfer?

Repeat the analysis on a second encoder and held-out intervention templates.

### Q4: Is it semantic rather than superficial?

Control for token count, edit distance, changed-token identity, sentence length, and intervention
family.

## Experimental units

The primary unit is a base text `x` with a verified intervention orbit:

```text
base_text
interventions:
  - family
  - expected_relation
  - strength
  - transformed_text
  - generator_id
  - verification_status
```

The pairwise unit is `(query, document)` with:

```text
query_id
document_id
relevance_label
collision_family
base_similarity
source
```

Generated examples must retain provenance. Unverified generations may be used for development but
not for the final evaluation set.

## Intervention set

The first version uses six families with matched controls.

| Family | Intended role | Critical example | Matched control |
| --- | --- | --- | --- |
| Surface | invariant | casing, punctuation, whitespace | unchanged canonical form |
| Lexical | usually invariant | verified synonym substitution | same token changed where relation is unaffected |
| Negation | critical | insertion, deletion, scope movement | negation word in a non-negating construction |
| Entity | conditional | subject or object replacement | irrelevant named-entity replacement |
| Quantity | conditional | number, unit, comparator change | formatting-only numeric change |
| Time/modality | conditional | date, tense, `may`/`must` change | stylistic auxiliary change |

Every critical family must include both relation-changing and relation-preserving cases. Otherwise a
model can solve the task by detecting the edit type.

## Scale

### Development run

- 2,000 base texts;
- 2 interventions per family;
- approximately 26,000 encoded texts including bases;
- one primary encoder;
- deterministic cache keyed by model revision, text hash, and normalization settings.

### Confirmatory run

- at least 10,000 base texts from multiple domains;
- at least 3 realizations for stochastic intervention families;
- both encoders;
- held-out templates and generator IDs;
- a human-verified collision subset sized before looking at final scores.

The confirmatory run should be frozen before final analysis.

## Measurement pipeline

For each base text:

1. encode and L2-normalize the base text;
2. encode each verified intervention;
3. compute tangent-projected response vectors;
4. group responses by semantic role and family;
5. build the weighted response matrix;
6. compute a deterministic truncated SVD;
7. save spectra, subspace sketches, scalar controls, and confounders;
8. discard raw texts from derived artifacts when dataset terms require it.

The SVD sign ambiguity must never enter a metric. Compare subspaces through projection matrices or
principal angles.

Uniform response-row weighting is the implemented default. Strength weighting is available only as
an explicit ablation through `--weighting strength`; it uses `sqrt(strength_i) * response_i`, so
squared response energy is linear in the frozen strength field. A strength-weighted result is not
interpretable on its own because relation-correlated strengths can inject the expected label.

Critical-versus-invariant energy is exported both as a stabilized log-ratio and as the bounded
contrast `(E_critical - E_invariant) / (E_critical + E_invariant + epsilon)`. The bounded contrast is
the preferred model feature when an encoder is exactly invariant to a control; the raw ratio must not
be used without its recorded stabilizer.

## Feature groups

### Base diagnostics

- embedding norm before normalization;
- cosine similarity for a pair;
- sentence length and token count;
- distance to a reference centroid;
- nearest-neighbour distance;
- local embedding density.

### Scalar perturbation diagnostics

- mean response norm;
- maximum response norm;
- response-norm variance;
- invariant-to-critical energy ratio.

### Spectral diagnostics

- top singular values;
- normalized spectral energy;
- effective response rank;
- spectral entropy;
- top-mode concentration;
- invariant/critical subspace principal angles;
- family-conditioned spectra;
- pairwise response-subspace compatibility.

## Prediction tasks

### Single-text tasks

- identify unstable examples under verified invariant transformations;
- predict whether an encoder will violate a known counterfactual contract;
- detect domain-shifted examples without calling the score calibrated OOD confidence.

### Pairwise tasks

- distinguish relevant pairs from semantic collisions matched by base cosine;
- predict top-K retrieval errors;
- rank which retrieved pairs should be sent to a cross-encoder;
- predict whether a critical edit changes the retrieval ordering as expected.

The pairwise tasks are primary. A text can be intrinsically valid yet collide with a particular
query, so a universal single-input score is insufficient.

## Statistical design

Split by source item and intervention generator, not by individual transformed sentence. This avoids
near-duplicate leakage.

Use:

- a development split for feature selection;
- a calibration split for thresholds or probability calibration;
- a locked test split for final reporting;
- bootstrap confidence intervals grouped by base text;
- at least three intervention-generation seeds where generation is stochastic.

Fit nested predictors:

1. confounders only;
2. confounders plus base diagnostics;
3. plus scalar perturbation diagnostics;
4. plus spectral and subspace diagnostics.

The relevant result is the incremental value of step 4.

## Primary evaluation

Report:

- collision AUROC and AUPRC;
- area under the risk-coverage curve;
- recall of true collisions at fixed review budgets;
- retrieval quality before and after selective reranking;
- calibration error only after explicit calibration;
- per-family performance;
- encoder calls and wall-clock cost.

Do not select a model using the same dataset on which the final claim is reported.

## Confounder falsification tests

The spectral hypothesis fails if its performance disappears when:

- base cosine similarity is matched;
- edit distance is matched;
- text and token length are matched;
- intervention-family identity is hidden or balanced;
- templates are held out;
- generated wording is replaced with human-authored examples;
- evaluation moves to the second frozen encoder.

Also run label-shuffle and response-row-shuffle controls. A spectral feature that survives label
shuffle indicates leakage or an analysis bug.

## Continuation gate

Advance to single-pass distillation only when:

1. spectral features improve collision AUROC by at least 0.03 over the best scalar perturbation
   baseline on the locked test set;
2. selective reranking reduces risk-coverage area by at least 10% relative without degrading full
   retrieval quality;
3. improvement appears in at least two critical-intervention families;
4. the second encoder shows the same direction of effect;
5. grouped confidence intervals exclude zero for the primary comparison;
6. human inspection does not reveal generator artifacts as the dominant signal.

Thresholds may be revised before data collection, but not after inspecting locked-test outcomes.

## Failure interpretations

Different failures imply different pivots:

- no signal in any response feature: abandon semantic response spectra;
- scalar variance works but spectrum does not: build a simpler stability estimator;
- spectrum works only for one edit family: narrow the product to that failure mode;
- spectrum works but not across encoders: treat it as model-specific monitoring;
- teacher works but cannot be distilled: use it as an offline audit or reranking tool;
- collision detection works but generic OOD does not: retain the pairwise objective and drop the OOD
  claim.

## First implementation slice

The initial code should implement only:

1. an encoder adapter with revision-aware caching;
2. typed deterministic interventions for surface, negation, entity, and quantity;
3. tangent response calculation;
4. truncated response SVD;
5. scalar and spectral feature export;
6. tests for normalization, tangent projection, row-permutation invariance, SVD invariance, caching,
   and split leakage.

No commutator, custom encoder, adaptive dimension policy, or end-to-end training belongs in the first
slice.

## Implemented Phase 0 runner

The repository now contains the numerical measurement core in `spectra_v3/` and a JSONL runner in
`experiments/phase0_measure_response.py`. Install the optional encoder dependency and run:

```bash
pip install -e ".[phase0]"
spectra-phase0 \
  --input examples/phase0_orbits.example.jsonl \
  --output work/phase0_features.jsonl \
  --cache work/phase0_embeddings.sqlite \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --revision <immutable-model-commit>
```

The revision argument is mandatory so a floating model update cannot silently change an experiment.
The runner writes a companion manifest with the input hash, encoder identity, measurement settings,
and cache size. Cache keys also include an inference fingerprint covering Torch and Sentence
Transformers versions, device, GPU model, and batch size. Raw texts are omitted from derived artifacts
unless `--include-text` is explicitly set. Each intervention retains its unweighted norm and the norm
used by the selected measurement weighting so matched controls can be compared within a family.
Existing outputs are protected unless `--overwrite` is supplied.

When the encoder exposes its tokenizer, the runner also exports token-ID edit distance, normalized
token edit distance, token-count delta, and token-ID Jaccard overlap for every intervention. These
are confounders, not semantic features: primary comparisons must survive matching or conditioning on
them.

An orbit may include `context_text`, normally a retrieval query or task statement. In that case the
runner exports the base query-document similarity and the signed and absolute similarity change for
every intervention. This query-conditioned response is required when expected preservation is
task-relative rather than a claim that the full document meaning is unchanged.

Spectral rank and energy apply a recorded numerical floor (`atol=1e-6`, `rtol=1e-5`) before summary
statistics are calculated. This removes float32 batch-layout noise from tokenizer-equivalent inputs.
Raw per-intervention response norms remain available so the effect of the floor can be audited.

For pipeline stress tests, `spectra-phase0-pilot` can generate a deterministic factorial corpus. It
contains 12 balanced interventions per orbit and writes `development_only: true` plus
`claim_eligible: false` in its manifest. Its purpose is debugging and confounder discovery; it cannot
support a research claim or replace the human-verified confirmatory set.

The implemented analysis tools also include:

- `spectra-phase0-summarize` for per-family confounder and response summaries;
- `spectra-phase0-late` for query-token/document-token late-interaction response;
- `spectra-phase0-fuse` for grouped and leave-one-family-out diagnostic fusion;
- `spectra-phase0-nli` for the frozen relation-aware NLI baseline.

The second development pilot found that these views are not interchangeable: late interaction
recovers entity, quantity, and time sensitivity but misses negation, while the frozen NLI baseline
solves the current synthetic template. The resulting typed-response revision and exact measurements
are recorded in [Phase 0 pilot v2 findings](PILOT_V2_FINDINGS.md).
