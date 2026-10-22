# Tracker width scaling

Scenario: [`cld_pf_heptv2_tracker_width_scaling_50k.yaml`](../configs/training/scenarios/cld_pf_heptv2_tracker_width_scaling_50k.yaml).

This study scales embedding dimension and backbone width together through 128,
256, 384 and 512, with seeds 12345, 23456 and 34567 for each width (12 jobs).
Depth stays fixed at six backbone layers and four decoder layers, with 256 slots.
Unlike the earlier model scaling study, width and depth are not changed together.

Every run starts from scratch with all parameters trainable. The common candidate
recipe combines hard hit pooling, the tracker momentum readout, and
particle-balanced tracker grouping CE. Grouping CE and background weights are
both 0.1. Sparse association, overlap, geometry, curvature and event-consistency
losses, tracker group readout and fit correction are disabled. The 154 baseline
input features retain tracker neighborhoods but exclude momentum moments and
doublets.

The tracker readout and balanced CE showed promising results separately; their
combination is not yet validated. This scenario measures scaling of that combined
recipe, not whether either component helps independently. The earlier loss and
readout factorials remain the controls for those questions.

All runs use global batch 128, LAMB at learning rate 0.001, cosine decay over
50,000 steps, validation and checkpoints every 1,000 steps, and activation
checkpointing. Validation and test budgets are 512 events per configured sample;
grouping diagnostics and the validation linear probe remain enabled.

Resolve the jobs without launching training:

```bash
uv run --no-sync python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cld_pf_heptv2_tracker_width_scaling_50k.yaml \
  --platform configs/training/platforms/lumi_mi250x.yaml --dry-run
```

Keep the hardware profile fixed: particle balancing is rank-local, so global
batch size alone does not fix its normalization population. The LUMI profile
uses eight GPUs with 16 events each. Compare matched checkpoints and evaluation
events, report all seeds and their spread, and measure charged momentum bias,
resolution and tails alongside efficiency/F1, grouping quality and neutral/jet
metrics. Report model parameter counts and compute cost as well as width.
