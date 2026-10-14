# Additional tracker-hit momentum features

Two opt-in input-feature blocks extend the current HEPTv2 hit model:

```yaml
model:
  hit_feature_engineering:
    tracker_neighborhood: true
    tracker_momentum: true
    tracker_doublets: true
```

Both new flags default to false. Existing checkpoints and the recommended model
retain their original input features. Enabling a block changes the input encoder
width and requires a new training run or explicit encoder-weight migration;
an old checkpoint cannot be resumed strictly with the new features enabled.
Neither flag enables the tracker-fit readout or adds a loss.

All features use detector inputs only, with zero outputs for calorimeter hits and
padding. They run in FP32 even during BF16 autocast. They can be evaluated offline
using the same modules, but this change does not add a disk cache or change TFDS.

## Neighborhood moments: 24 extra channels

`tracker_momentum` extends `TrackerNeighborhoodFeatures` and requires the existing
`tracker_neighborhood` block. It reuses its small/large angular bins (0.025/0.1),
surface counts, conformal centroids and covariance axes.

- Two smoothly compressed coordinates: `asinh(100*x/rho²)` and
  `asinh(100*y/rho²)`, with positions in mm. These supplement the original clipped
  geometry features without overwriting them. At zero radius they are zero.
- Eleven channels at each angular scale: signed conformal line offset, compressed
  curvature magnitude, cosine/sine of twice the principal-axis angle, along-axis
  standard deviation, relative transverse RMS, curvature eligibility, compressed
  longitudinal slope, longitudinal intercept, longitudinal residual RMS and
  longitudinal eligibility.

For the conformal coordinates `u=100*x/rho²`, `v=100*y/rho²`, the principal-axis
normal `n` gives offset `d=n·mean(u,v)`. A circle through the origin has
`|curvature| = 2*|d|/100` in inverse mm. The input channel is
`asinh(2000*|d|/100)`, i.e. compression of curvature in inverse metres. The signed
offset follows the PCA axis convention; it is **not** a charge estimate.
The existing neighborhood moments retain their existing ±8 conformal clipping.

Curvature is eligible with at least three distinct surfaces, radial span at least
100 mm and nondegenerate along-axis variance. Offset, curvature and axis channels
are zero otherwise. Eligibility indicates geometric support, not particle purity;
the residual and span features let the network assess reliability.

The longitudinal fit is an inexpensive `z = slope*rho + intercept` regression,
not a helix fit. It requires two surfaces, at least 5 mm radial span and nonzero
radial variance. Slope uses `asinh`; intercept and residual use a 3000 mm scale.
Four longitudinal moments share the existing scatter reduction. Direct centered residual sums
avoid unstable subtraction of nearly equal variances for straight tracks.

## Bounded doublets: 12 extra channels

`TrackerDoubletFeatures` in `mlpf/model/tracker_features.py` packs tracker hits,
sorts them into an event-local angular grid and searches nine neighboring bins.
Phi wraps periodically. Each bin contributes at most 16 candidates sampled across
radial order; query chunks of 2048 bound temporary pair storage. There is no
all-hit-pairs matrix. Sorting scales as O(N log N), followed by bounded work per
tracker hit, and the zero-padded output allocation scales with total hit count.

For each hit, the nearest angular candidate in each radial direction must be on a
different detector surface, separated by at least 5 mm radially and within
`deltaR <= 0.15`. These are cross-surface neighbors, not necessarily consecutive
detector layers. The candidate search is deliberately capped and does not promise
the globally nearest hit. Canonical sorting handles equal-radius ties.

Each direction contributes validity, compressed signed transverse curvature,
compressed `delta_z/delta_rho`, radial span / 3000 mm, angular separation and
`log1p(eligible candidate count)`. The curvature comes from the origin and the two
hits, ordered from inner to outer radius:

```
curvature [1/m] = 2000 * cross(xy_inner, xy_outer)
                 / (rho_inner * rho_outer * distance_xy)
```

Its sign describes transverse bending in this coordinate convention. Missing
neighbors produce zeros, including validity. Small-angle searches can miss highly
curved tracks; mixed-particle pairs can still pass. These are hints for the learned
momentum head, not replacements for it or for material-aware tracking.

## Controlled training scenario

`configs/training/scenarios/cld_pf_heptv2_tracker_momentum_50k.yaml` has three
variants: unchanged tracker-feature baseline, added moments, and added moments
plus doublets. All use the current grouped HEPTv2 model and activation
checkpointing, global batch 512, 50k steps, seed 12345, LAMB and LR 0.001 with
cosine decay. The permitted variant differences are only the two feature flags.
Input dimensions are respectively **154, 178 and 190**. The extra input channels
also slightly increase the encoder parameter count.

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cld_pf_heptv2_tracker_momentum_50k.yaml \
  --platform configs/training/platforms/lumi_mi250x.yaml --dry-run

GLOBAL_BATCH_SIZE=4 NUM_STEPS=4 VAL_FREQ=2 CHECKPOINT_FREQ=2 \
NVALID=4 NTEST=4 NUM_WORKERS=0 VALIDATION_DIAGNOSTICS_BATCHES=1 \
  bash scripts/local/train_scenario.sh cld_pf_heptv2_tracker_momentum_50k \
  --variant hit_heptv2_tracker_momentum_doublets
```

As with the earlier batch-512 scenario, the full batch of 64 events per device
still needs target-platform memory/ROCm/DDP validation.

## Local tests and cost, 2026-10-14

RTX 5060 Ti 16 GiB, PyTorch 2.11/CUDA, BF16, four real CLD ttbar hit events per
batch. The fixed feature benchmark contains 11,904 padded hits/event; ten timed
calls follow three warmups. Full training uses five different real batches and
averages the last three steps, including the set/grouping losses and LAMB update.

| Variant | Feature construction | Full training step | Peak allocated during training |
|---|---:|---:|---:|
| Baseline | 11.98 ms | 749.9 ms | 3171.0 MiB |
| Moments | 14.42 ms | 758.0 ms | 3178.8 MiB |
| Moments + doublets | 30.52 ms | 788.4 ms | 3181.5 MiB |

In these short measurements, moments add about 1.1% to training-step time and
doublets bring the total overhead to 5.1%; allocated memory rises by under 0.4%.
This is a local cost measurement, not a convergence or momentum-resolution result.

The CPU regression suite passed 105 tests (three GPU tests skipped); the focused
GPU suite passed 13 tests. Tests cover analytic circle curvature through a 1000 m
radius, both bending signs, longitudinal slope/intercept, noisy-neighborhood
quality, missing neighbors, degenerate inputs, event isolation, padding, candidate
caps, permutation invariance, the phi seam and BF16-autocast CPU/GPU agreement.
The new channels use tighter comparison tolerances than legacy circular-width
features, which exhibit FP32 cancellation in nearly identical angular inputs.

All three variants completed real-data loss/backward/LAMB and validation checks
with finite gradients. The normal launcher also completed four steps with both
blocks enabled, validation diagnostics and predictions at steps two/four, and
checkpoint saving. Outputs are under `experiments/tracker_momentum_features_local/`.

Reproduce cost measurements with `scripts/studies/benchmark_heptv2_checkpointing.py`:
use `--batch-size 4 --checkpointing`, adding `--tracker-momentum` and optionally
`--tracker-doublets`; add `--features-only` for the fixed-batch feature benchmark.
Use a separate `--output` JSON for each mode and avoid competing GPU workloads.

The next physics test is the full ablation: charged momentum core widths and
tails versus pT/eta, together with particle-finding F1 and jet/MET metrics. A
short smoke test does not establish that these features improve reconstruction.
