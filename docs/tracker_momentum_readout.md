# Tracker momentum readout and sparse association

`cld_pf_heptv2_tracker_readout_50k` is a four-way ablation based on the small
model in `cld_pf_heptv2_model_scaling_50k`, at revision `da748c79`.

| Variant | Tracker momentum readout | Sparse differentiable pooling |
| --- | --- | --- |
| `hit_heptv2_small_baseline` | off | off |
| `hit_heptv2_tracker_readout` | on | off |
| `hit_heptv2_sparse_association` | off | on |
| `hit_heptv2_tracker_readout_sparse` | on | on |

All use width 256, six HEPTv2 layers, four decoder layers, 256 slots, the previous
154-channel engineered features, and backbone activation checkpointing. The
recipe keeps global batch 128, seed 12345, LAMB, LR 0.001, 50k cosine steps,
and validation/checkpoint intervals of 1,000. No new input moments/doublets,
legacy circle-fit anchors, curvature loss, or event-consistency loss are enabled.
The scenario resolver permits differences only in the two feature flags.

## Sparse association

The existing initial query-to-hit affinity projections and supervised grouping
loss are retained. For each hit, select four slots by the existing cosine score
(scale 10). Selection is discrete and runs without autograd, in chunks of 1,024
hits. Recompute the selected dot products with gradients and normalize those
four scores **plus background** with a temperature-1 softmax. This is a truncated
softmax; it does not preserve the mass of the original full-slot distribution.
Weights are not renormalized after dropping background.

Weighted detector-specific embedding means, effective counts, and deposited
calorimeter energy replace hard membership in the grouping readout. Accumulation
is FP32. The denominator is clamped to one so nearly empty groups are suppressed,
not amplified. Momentum and other downstream losses can update selected hit/slot
scores and background scores; top-k membership itself has no derivative. The
full supervised grouping objective still trains affinities outside the selected
support. Existing ownership diagnostics continue to describe hard argmax labels,
not the fractional memberships used by this branch.

The temporary selection scores are chunk-by-slots; differentiable selected
scores scale as hits-by-k. No full event hit-by-slot affinity matrix is retained
for pooling backward. This reduces storage, not the O(hits x slots) candidate
search cost. Decoder attention and supervised grouping retain their existing costs.
Association is computed once from initial proposals, as in the baseline; it is
not iteratively updated after every decoder layer.

## Tracker-specific momentum readout

Only tracker hits contribute directly to this branch. Each slot receives its
weighted mean tracker embedding and four radial-bin geometric summaries. The
radial intervals are [0,0.1], (0.1,0.3], (0.3,0.7], and (0.7,infinity) metres.
These are CLD-oriented radial bins, not actual detector-layer identifiers.
Coordinates are read from raw-hit columns 6–8 (millimetres) and converted to metres.
Each bin stores weighted means of x, y, z, r, regularized conformal u and v,
x², y², xy and rz, plus log(1 + effective count). Nonfinite positions are excluded
from the geometric summaries. Conformal coordinates are 0.1*x/max(r²,1e-6) and
the corresponding expression for y. Geometry and reductions remain FP32.

A separate embedding LayerNorm, hidden GELU layer and zero-initialized output
predict a bounded additive log-pT correction (maximum absolute value log(1.5)).
The correction has a continuous effective-tracker-count support gate, clamped
at one, and is exactly zero without tracker support. It applies only when the
predicted physical PID is charged (classes 1,4,5), using the same hard PID choice
as the legacy fit-correction branch. PID selection receives no gradient from this
gate. Energy, direction, presence and PID heads are unchanged. The new correction
is applied to each supervised decoder output, using the same initial-group summary.

The complete head and final pT addition run in FP32 under mixed precision.
At initialization the correction is exactly zero; upstream gradients through
the head start once its output weights move away from zero. The ordinary scale
head remains trainable and provides fallback predictions. This is a learned
tracker residual, not an analytic circle fit or a calibrated uncertainty estimate.
Although the pooled inputs are tracker-only, their backbone embeddings can contain
calorimeter context. The readout-only variant preserves baseline hard pooling;
only the sparse variants change its accumulation and association behavior.

Both options default off and add no baseline parameters when disabled. Old model
state dictionaries load unchanged with defaults. Enabling a readout on an old
checkpoint introduces new parameters and requires an explicit warm-start policy;
the supplied experiment starts each variant from scratch. Combining these options
with legacy `tracker_group_readout`/fit anchors is rejected rather than silently
changing their meaning. New fields are validated by `SetDecoderConfig`.

## Running and validation

Preview all four jobs without submitting training:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cld_pf_heptv2_tracker_readout_50k.yaml \
  --platform configs/training/platforms/lumi_mi250x.yaml --dry-run
```

Launch on LUMI with:

```bash
scripts/lumi/train_scenario.sh cld_pf_heptv2_tracker_readout_50k mi250x
```

Tests check dense-reference weights and pooling gradients, background rejection,
chunk/order/padding invariance, empty events, neutral/no-tracker fallback,
zero-initialization compatibility, BF16 forward/backward and scenario invariants.
Completed 50k physics results are summarized in [the evidence summary](cld_hit_results.md):
the hard-pooled head improves momentum widths/tails modestly, while sparse
variants regress. Isolated target-device throughput and inference memory still
need measurement. Compare at common checkpoints, including charged-hadron width,
efficiency and tails together; the earlier fit study showed that width alone can
improve while reconstruction quality deteriorates.

Implementation validation: 102 targeted tests passed across readout, set prediction,
grouped tracking and scenario resolution. All four full-size models passed a CPU
BF16 forward/backward check on 128 synthetic hits, with finite outputs and
gradients. Baseline and sparse-only models have 12,709,139 trainable parameters;
readout variants have 12,786,964. All four LUMI commands passed scenario dry-run
resolution. These checks do not measure realistic GPU occupancy or physics quality.
