# Small-model tracker grouping loss ablation

Scenario: [`cld_pf_heptv2_tracker_grouping_losses_50k.yaml`](../configs/training/scenarios/cld_pf_heptv2_tracker_grouping_losses_50k.yaml).

This four-arm factorial tests whether more direct supervision of charged-particle
hit groups improves momentum reconstruction. Every arm starts from scratch
(`load: null`, all parameters trainable), using the same small HEPTv2 architecture:
width 256, six encoder layers, four decoder layers and the baseline engineered
hit features. Hard pooling and the baseline momentum readout remain fixed.

| Arm | Particle-balanced tracker CE | Tracker overlap weight |
| --- | --- | --- |
| `hit_heptv2_small_baseline` | Off | 0 |
| `hit_heptv2_tracker_particle_balanced` | On | 0 |
| `hit_heptv2_tracker_overlap` | Off | 0.05 |
| `hit_heptv2_tracker_particle_balanced_overlap` | On | 0.05 |

All arms use seed 12345, global batch size 128, LAMB with learning rate 0.001,
cosine decay over 50,000 steps, and validation/checkpoints every 1,000 steps.
The existing grouping CE coefficient is 0.1 and background weight is 0.1.
Geometry, curvature, and event-consistency losses are disabled, as are the
optional tracker readouts and sparse pooling. The factorial changes only the
two loss settings above.

## Loss definitions

Both losses use the final Hungarian assignment to map truth particles to slots.
Charged truth classes are charged hadrons, electrons and muons (IDs 1, 4, 5).
Eligibility depends on truth labels and tracker-hit support, not predicted PID,
hard assignment, confidence, or successful geometry fits.

**Particle-balanced tracker CE** replaces the unit weight of each charged truth
tracker hit in track j with `N / (K * n_j)`, where `n_j` is its truth hit count,
`N` is the total number of eligible hits and `K` the number of supported charged
tracks in the rank-local batch. Each track consequently receives equal total
weight, while the total charged-tracker CE weight is preserved. Calorimeter,
background, and other tracker-hit weights are unchanged. The existing CE
normalization is retained. This is rank-local balancing, not a global distributed
track-count reduction; keep the hardware profile fixed across comparisons.

**Soft tracker overlap** adds the mean per-track soft Dice loss:

```
TP_j = sum over truth tracker hits of particle j: p(hit -> matched slot j)
M_j  = sum over all valid tracker hits: p(hit -> matched slot j)
T_j  = number of truth tracker hits of particle j
L_overlap = mean_j [1 - (2 TP_j + epsilon) / (T_j + M_j + epsilon)]
```

Probabilities come from the full slot-plus-background softmax. Thus foreign and
background hits contribute to predicted mass, and missed truth hits still
receive supervision. There is no top-k or predicted-membership gate.
Only matched charged truth particles with tracker support enter the average.
The initial coefficient 0.05 is an experimental choice, applied from step one
without a warmup; it has not been optimized.

The overlap calculation uses FP32 probabilities and reductions with chunked
checkpoint recomputation. It introduces training cost but no model parameters or
inference computation. Both options default off for existing configurations.

## Running and comparing

Resolve and inspect the four jobs:

```bash
uv run --no-sync python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cld_pf_heptv2_tracker_grouping_losses_50k.yaml \
  --platform configs/training/platforms/lumi_mi250x.yaml --dry-run
```

The generic runner executes jobs on the current machine; it does not allocate
a Slurm node. To submit on LUMI, use
`scripts/lumi/train_scenario.sh cld_pf_heptv2_tracker_grouping_losses_50k mi250x`.
Global batch 128 resolves to 16 events per GPU with this eight-GPU profile.
Partial matched-32k results are in [the evidence summary](cld_hit_results.md).

Compare all four arms at identical steps and on identical evaluation events.
Use charged-hadron momentum bias, resolution and tails alongside particle F1,
efficiency/fake rate, and tracker grouping purity/completeness. Inspect low-hit
tracks and momentum/eta bins to determine which particles benefit. Preserve jet,
MET and neutral-particle metrics as cross-checks.

The overlap arms additionally log `TrackerOverlap`, `TrackerOverlapCount`,
`TrackerSoftPurity`, and `TrackerSoftCompleteness`. Existing hard-group validation
diagnostics are enabled for every arm. Total training loss is not directly
comparable across arms because the optimized objectives differ. A grouping gain
without a momentum gain would motivate studying how the readout uses the groups;
a momentum gain accompanied by lower reconstruction efficiency is a tradeoff,
not an unqualified improvement.

One seed is an initial screening experiment. Repeat promising arms with more
seeds before drawing a firm conclusion.
