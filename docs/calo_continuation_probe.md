# Calorimeter cross-layer continuation probe

This follow-up tests particle identity in ambiguous shower neighborhoods rather
than binary edge AP on a graph dominated by same-particle pairs. It reuses the
15 raw features and 256 frozen HEPTv2 embedding channels cached by
`scripts/studies/probe_hit_instances.py` at the completed grouped 50k checkpoint.
The backbone is never updated. Four scores are compared: spatial distance,
directional distance, a linear raw-feature ranker, and a linear embedding ranker.

## Candidate construction and triplets

For each foreground calorimeter hit, find hits on layer `layer + 1` in the same
`system` and `side`. This refers to the next numbered detector layer, not the
next layer with a hit. No links cross detector systems or sides. On that target
layer, union the 16 nearest hits in xyz with the 16 nearest in
(eta, sin(phi), cos(phi)). This graph uses geometry only and is identical for
all four scores. Unlike the original graph, these are directed continuations.

Use particle_number only to label candidates. For each anchor, enumerate
positive/negative pairs with the same/different particle_number respectively.
Require BOTH spatial and directional distances to satisfy
`max(d_positive, d_negative) / min(d_positive, d_negative) <= 1.25`, with numerical
floors of 1e-6 mm and 1e-8 respectively in the denominators. Sample one admissible
pair uniformly per anchor using a fixed event-dependent seed. The test is thus
conditioned on the existence of a competing particle at a comparable distance.
Triplet sampling uses truth, as a supervised diagnostic, rather than an
inference-time algorithm. Geometry-only candidate construction remains separate.

## Fitting and model selection

The pair score is `w dot standardize(abs(h_anchor - h_candidate))`. Train using
`log(1 + exp(-(score_positive - score_negative)))`, implemented as symmetric
positive/negative logistic examples with no intercept (a common score intercept
would cancel). Standardization uses training pairs only.

Each represented particle receives equal weight within its event, then its
eligible anchors share that weight. Events receive equal total training weight.
This also defines the reported ranking and retrieval accuracy: mean over
particles within each event, then mean over events. Coverage counts additionally
report support in hits/anchors, so conditional results are explicit.

Fit L2-regularized logistic regressions to convergence. Select inverse
regularization C from 0.01, 0.1, 1, 10 using validation triplet accuracy only,
separately for raw features and embeddings. Evaluate once on the held-out events.
The inherited 32/8/16 event splits are unchanged and event overlap is rejected.

## Evaluation

- Distance-matched triplet accuracy: does the correct continuation score higher
  than the competing particle? Exact ties receive half credit. Chance is 50%.
- Top-1 retrieval on the complete candidate sets, separately for anchors with
  both classes present, anchors with a positive available, and all anchors with
  a target layer. Exact top-score ties receive fractional credit.
- Coverage: total foreground hits, anchors with a target layer, anchors whose
  particle exists in that layer, anchors with a positive in the graph, anchors
  with competing candidates, and anchors with an admissible triplet.

Top-1 retrieval uses the complete geometry-only candidates, not just the sampled
triplet, and remains distinct from instance segmentation. These rankers do not
produce particle clusters or solve background rejection. Input hits still use
an oracle foreground mask. The TFDS test split was excluded from backbone
updates but may have contributed to backbone validation/model selection.

## Run locally

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run python \
  scripts/studies/probe_calo_continuation.py \
  --source experiments/hit_instance_probe_50k_initial \
  --output experiments/calo_continuation_probe_50k_repeat
```

A fresh output directory is required. CPU fitting suffices because frozen
embeddings are already cached. Outputs include the inherited provenance,
SHA256 hashes of every input cache, exact candidate lists and triplets,
validation sweeps, fitted probes and scalers, and per-event test results.
`--layer-gap` permits larger layer separations; `--distance-ratio` controls
matching tolerance. Use a validation-driven protocol for further tuning.

## Initial result (2026-10-12)

Completed locally using the existing 32/8/16 ttbar split, with adjacent numbered
layers and a distance ratio limit of 1.25. There are 43,220 training triplets,
11,839 validation triplets, and 29,728 test triplets. All fitted models converged.
Validation selected C=1 for raw features and C=0.01 for embeddings.

| Score | Distance-matched triplet accuracy | Top-1, complete competing candidate set |
|---|---:|---:|
| Spatial distance, untrained | 60.52% | 97.18% |
| Directional distance, untrained | 59.42% | 97.03% |
| Raw-feature linear ranker | 58.44% | 94.09% |
| Frozen-embedding linear ranker | **67.73%** | 91.72% |

Embeddings improve the primary triplet metric by **9.29 percentage points** over
raw features and 7.22 points over spatial distance. A paired bootstrap over the
16 test events (10,000 resamples, seed 12345) gives a 95% interval of +6.85 to
+11.61 points for the embedding-minus-raw triplet gain. This captures event
sampling variation, not training-seed uncertainty.

The direction of the result reverses for unrestricted top-1 retrieval. Most
anchors still have an easy nearest same-particle continuation, which spatial
distance finds reliably. Fitting a ranker to ambiguous triplets does not make it
a better general neighbor selector. The 25% matching tolerance also leaves
measurable geometric information: untrained distance scores exceed 50% on the
triplets. Thus the result supports additional particle-identity information in
the frozen embeddings, not complete removal of geometric discrimination.

Coverage across all test foreground hits (pooled counts):

- 98.65% have any hits on the next layer in the same system and side.
- 92.72% have their particle represented on that next layer.
- 92.41% have at least one positive in the candidate graph.
- 83.52% have both positive and negative candidates.
- 21.44% support a triplet matched in both distance measures.

The principal result is conditional on that last subset. Each metric then
weights represented particles equally within events, rather than weighting all
hits equally. A particle can contribute a different set of eligible anchors to
each metric. Output artifacts are in `experiments/calo_continuation_probe_50k/`,
including the result JSON, fitted probes, exact candidate/triplet samples,
provenance, and `comparison.png`.

## During training validation

Enable the bounded online diagnostic with:

```yaml
validation_diagnostics_batches: 4
validation_linear_probe: true
validation_linear_probe_max_events: 16
```

It is enabled for all variants of `cld_pf_heptv2_grouped_tracking_50k` for future
launches. Existing running jobs do not pick up code/config changes automatically.
It applies to set-output hit models and uses only the configured diagnostic
batches. A temporary decoder pre-hook captures the existing backbone output,
so no extra backbone forward is needed. Only detached foreground hit
features leave the device. A maximum of 16 events per rank is retained on CPU.

Repeated records are deduplicated by raw-feature and truth-label content.
Every fourth collected event is held out; the others fit fresh probes at each
validation. The default cap gives 12 fit / 4 evaluation events per rank if enough
calorimeter events are available. At least four fit events with triplets and two
evaluation events with triplets are required. With local batch size one, increase
`validation_diagnostics_batches` to at least eight to obtain a fit. The configured
maximum limits event count, not hit count. Memory and runtime depend on occupancy.

Online probes use the same adjacent-layer graph, 25% distance matching and
particle weighting as the offline study. They fix C=0.1 for both representations,
with at most 200 solver iterations and one CPU thread, instead of performing a
hyperparameter sweep. If either probe fails to converge, learned accuracies are
omitted and `fit_success_fraction` records the unsuccessful rank. Geometry
baselines and coverage remain available when there are evaluation events.
No weights or scalers persist between validation cycles. These diagnostic fits
never update backbone parameters or contribute to training/validation losses.

Under DDP, each rank fits its own small probes. Accuracies are reduced as
sums/counts of evaluation-event particle-macro accuracies. They describe an
ensemble of rank-local diagnostics, not a single probe fitted to global events.
Coverage ratios reduce pooled hit counts. Support names ending in `samples`
report global sums. `seconds_per_rank` is mean CPU collection/fitting/evaluation
time, not full validation time. Static metric keys keep collectives aligned
when a rank has no usable examples. A rank with an unsuccessful learned fit
does not contribute to learned-accuracy denominators.

TensorBoard tags start with `validation/linear_probe/calorimeter/`:

- `{raw,embedding,xyz_distance,direction_distance}/triplet_accuracy`
- The same methods' `top1_competing` and `top1_all_candidate_anchors`
- `embedding_minus_raw/triplet_accuracy`
- `coverage/{target_layer_exists,same_particle_in_target_layer,positive_in_candidates,competing_candidates,distance_matched_triplets}`
- `fit_success_fraction`, `fit_event_samples`, `eval_event_samples`,
  `fit_triplet_samples`, `eval_triplet_samples`, `collected_event_samples`,
  and `seconds_per_rank`

The same values appear in `history/step_*.json` under
`valid.metrics/linear_probe/calorimeter/...`, and in rank-zero log lines prefixed
`VALIDATION_PROBE | Step=...`. Compare trends within this fixed protocol; the
small rank-local fits have different sample sizes and regularization from the
offline 32/8/16-event benchmark.


### Instance ARI in TensorBoard and logs

The same switch now also enables separate tracker and calorimeter instance
probes. These use the original within-detector union of 16 xyz and 16 directional
neighbors, rather than the cross-layer continuation graph. Both representations
share exact edge samples and event splits. Fresh logistic affinity classifiers
fit at most 2048 candidate edges per fitting event, with balanced class weights
and equal event weight, fixed C=0.1 and at most 200 iterations. Standardization
uses fitting events only. No checkpoint weights receive gradients.

For ARI, event positions modulo four select **fit, fit, calibration, evaluation**.
The default 16-event cap therefore gives 8/4/4 per rank. At least 4/2/2 events
are required. Calibration selects the edge-probability threshold separately for
each representation from 0.30, 0.50, 0.70, 0.85, 0.95, 0.98, 0.995 by mean ARI.
Final evaluation events never affect probe fitting or threshold selection.
Connected components of retained edges give instances; singleton hits remain.
The scalar is mean per-event adjusted Rand index, not a particle-weighted mean.
As elsewhere, these diagnostics use truth-associated foreground hits only.

New tags under `validation/linear_probe/{tracker,calorimeter}/`:

- `raw/ari` and `embedding/ari`: held-out instance agreement.
- `embedding_minus_raw/ari`: paired per-event improvement.
- `oracle_graph/ari`: perfect same-particle classification on candidate edges.
- `raw/ari_threshold`, `embedding/ari_threshold`: mean selected threshold across
  successful ranks (each rank can select a different threshold).
- `ari_fit_event_samples`, `ari_calibration_event_samples`, `ari_eval_event_samples`:
  global support counts, with no scalar accuracy emitted for unsuccessful fits.
- `ari_fit_success_fraction`: fraction of ranks with usable fits/calibration.

All these also appear in history JSON with a `metrics/` prefix and in
`VALIDATION_PROBE` log lines. DDP aggregates event sums/counts, while fitting and
threshold calibration remain independent on each rank. The online estimator
shares the slides' graph/partition definition but uses a smaller fitting sample
and a different optimizer, so compare trends rather than expecting identical
values. `seconds_per_rank` now includes both detectors' ARI and continuation work.
