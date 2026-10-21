# Tracker-fit supervision experiments

The controlled experiment is `cld_pf_heptv2_fit_loss_50k`. It builds on
`pyg-cld-hits-heptv2-set-v1`, the current grouped HEPTv2 model with tracker
neighborhood features. It adds training-only supervision of learned ownership;
all four variants keep the existing learned momentum, energy and direction
readouts. No additional network parameters or inference operations are added.

## Motivation and controls

The frozen filtering study in `experiments/local_frozen_tracker_filtering_20261013`
found that combined confidence and geometric filtering narrowed charged-particle
pT residuals from 15.14% to 12.38% on its held-out sample, with tails approximately
unchanged. Geometry alone increased tails. This motivates teaching ownership
which hits are compatible with a trajectory before changing the momentum readout.
These frozen results are motivation, not results for this new training suite.

| Variant | Geometry weight | Curvature weight |
|---|---:|---:|
| `tracker_features_baseline` | 0 | 0 |
| `tracker_geometry` | 0.01 | 0 |
| `tracker_curvature` | 0 | 0.1 |
| `tracker_geometry_curvature` | 0.01 | 0.1 |

Each uses seed 12345, 50,000 steps, batch 64, learning rate 0.001 and the existing
validation schedule. Train from the same initialization recipe; these are not
fine-tunes of the 50k checkpoint. Loss weights are initial experimental choices,
not calibrated optima. The current default remains unchanged.

## Implemented objective

The ordinary particle losses, Hungarian matching and supervised hit ownership
cross-entropy remain in place. Only matched charged hadrons and muons receive
these new losses; electrons are excluded for this initial material-free model.

Candidate selection uses the model's own hard assignment, including its background
class. At most 64 tracker hits per slot are retained, ranked by ownership
probability. Selection is detached and does not use truth hit ownership. Within
this subset, weights are recomputed as the differentiable softmax probability of
the matched slot against every other slot and background.

**Geometry:** Fit a detached reference circle to the truth-owned tracker hits of
the matched particle. Require at least four hits, three distinct detector
surfaces, 10 cm radial span, adequate conditioning and transverse RMS below 1 mm.
Minimize the ownership-weighted mean smooth-L1 distance of candidate hits to this
reference circle, in units of 1 mm. This is supervised compatibility with a truth
trajectory, not a self-consistency loss that could accept the wrong circle.
Truth hit ownership is used only to construct this training target.

**Curvature:** Fit a circle to the model-selected hits using continuous learned
weights. A normalized double-precision regularized 3-by-3 least-squares solve is
differentiable with respect to these weights. Supervise inverse pT using
`1 / (0.299792458 * B * radius)` against truth inverse pT, with smooth-L1 beta 0.2.
The residual is divided by `max(1 / truth_pT, 0.02 / GeV)`; above 50 GeV this
becomes an absolute inverse-pT error rather than an increasingly amplified
relative error. This avoids differentiating pT as curvature approaches zero.

Predicted fits require the same minimum hit/surface/span constraints, Gram
minimum/maximum eigenvalue ratio above 1e-7 and effective weighted hit count of
at least four. Gates are detached. There is deliberately no residual-quality
cut on the predicted circle: otherwise a contaminated but well-conditioned fit
could simply be excluded from momentum supervision. Empty eligible samples give
a differentiable zero. Losses are averaged over eligible particles in each batch.

`TrackerGeometry`, `TrackerCurvature` report weighted auxiliary losses.
`TrackerCandidateCount`, `TrackerGeometryCount`, `TrackerCurvatureCount` are
per-batch diagnostic counts, excluded from the optimized total. Monitor counts
alongside losses: declining coverage must not masquerade as better fitting.

## Cost and limitations

Assignment selection is chunked; softmax activations for each capped candidate
set are checkpointed. There is no all-hit-pairs matrix. Work still scales with
hits times slots, plus candidate hits times slots for matched charged particles.
The first implementation loops over matched particles and uses detached GPU
quality decisions, so synchronization overhead needs measurement at production
batch size. The candidate cap bounds per-track work, not total batch memory.

This is a transverse circle experiment, not a full material-aware helix fit.
A 1 mm reference tolerance is an initial choice, not a detector covariance model.
Truth circle rejection may select easier tracks. Hard candidate selection cannot
recover omitted hits through the new loss; ownership CE still supervises them.
Normalized weighted losses can concentrate on a subset or lose eligibility; CE
and the effective-count gate help, but do not mathematically prevent collapse.
Track fit coverage and ownership completeness must therefore remain acceptance
criteria. Short or nearly straight tracks can fail the conditioning cut.
The new weighted fit is not the production detached IRLS readout.

## Execution and decision criteria

Resolve the full suite without launching training:

```bash
.venv/bin/python scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cld_pf_heptv2_fit_loss_50k.yaml \
  --platform configs/training/platforms/local.yaml \
  --global-batch-size 1 --dry-run
```

For a short local training pilot, run each variant through the existing launcher:

```bash
NUM_STEPS=2000 VAL_FREQ=500 CHECKPOINT_FREQ=500 \
  scripts/local/train_scenario.sh cld_pf_heptv2_fit_loss_50k \
  --variant tracker_geometry_curvature
```

Repeat for the other three variant names. Local batch defaults to one; this is a
pilot with its own paired baseline, not a batch-64 comparison. Resolve the same
scenario against the intended cluster platform for the full batch-64 suite.
The launch examples describe the configured recipe. The archived campaign has
partial results; see [the dated evidence summary](cld_hit_results.md).

At common steps compare charged-hadron and muon pT central 68% half-width, bias,
MAE and >50% tails by truth pT and eta, plus charged efficiency, neutral metrics,
particle F1, jet response IQR/F1 and event momentum closure. Report fit coverage,
retained-hit purity/completeness and processing time/memory. Use a fixed matching
cohort as well as the normal rematched metrics, to expose matching selection.
Do not compare total training losses across variants with different objectives.

Use validation data for selection and reserve test data for the chosen setup.
Suggested guards are no more than +0.1 percentage point charged tail rate and
-0.2 percentage point charged efficiency versus the paired baseline, with paired
event bootstrap uncertainty. Do not promote a model based only on a narrower
core when jet/event metrics or fit coverage deteriorate. Confirm a winner with
additional seeds before making it the default.

The separate [bounded fit-correction experiment](charged_fit_correction.md) has
also been evaluated and did not establish an overall reconstruction improvement. Longitudinal
helix consistency and a broader candidate pool are separate follow-ups so that
this first suite isolates the effect of fitting supervision.

## Initial implementation validation

The targeted fit-loss, set-prediction and scenario tests passed (87 tests),
including finite-difference checking of the circle's weight gradient and a
contamination example where gradient descent lowers the outlier ownership score.
The four scenario commands resolve successfully in a dry-run.

A local BF16 GPU smoke check used the frozen winning 50k checkpoint and the first
test event from each of ttbar, WW and qq. It backpropagated only through the new
losses into detached ownership representations, without optimizer updates.
Geometry/curvature eligible counts were 17/42, 28/42 and 8/13 respectively;
all three ownership tensors received finite, nonzero gradients. Loss-only
forward/backward took 0.37, 0.18 and 0.06 seconds respectively. This is not an
end-to-end training memory or throughput benchmark. Results are saved in
`experiments/local_tracker_fit_loss_smoke_20261013.json`.
