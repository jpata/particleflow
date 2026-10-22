# HEPTv2 activation checkpointing and reduced-batch training

`model.heptv2.activation_checkpointing=true` checkpoints each HEPTv2 backbone
layer in `MLPF._run_backbone`, as proposed in
[issue #520](https://github.com/jpata/particleflow/issues/520).
It uses non-reentrant PyTorch checkpointing with RNG preservation. Only training
with gradients enabled takes this path. The default is false; inference, parameter
names, saved weights, hashing, bucket size, decoder and training objective are
unchanged. Backward recomputes layer activations instead of retaining them.

## Training recipe

`configs/training/scenarios/cld_pf_heptv2_batch512_50k.yaml` contains two variants:

- `track_cluster_mlpf`: the six-layer track/cluster reference.
- `hit_heptv2_grouped_tracker_features`: the current recommended six-layer HEPTv2
  model with the four-layer global decoder, hit grouping and tracker-neighborhood
  features, plus activation checkpointing. No track-fitting branch is enabled.

Both use global batch **384**, seed **12345**, **50,000 optimizer steps**, LAMB,
learning rate **0.001**, weight decay **0.01**, and the existing cosine schedule
(ending at 10% of the initial LR). Validation and checkpointing run every 5,000
steps, with `nvalid=ntest=512`, interleaved-shard sampling and padding to 128 hits.
The eight-device profiles resolve to **48 events per device**, without gradient
accumulation. This gives **19.2 million event presentations**, reduced from
September's 25.6 million to provide GPU memory headroom. The historical scenario
name is retained for existing launch commands.
The BF16 hardware profiles retain their existing runtime settings.

Both variants share the optimization budget. The hit model uses raw hits, set
matching, auxiliary decoder and grouping losses; the track/cluster model uses
elementwise targets. Hardware and attention backend
also differ between the H100 and MI250X profiles. The existing batch-64 default
scenario remains unchanged.

The new scenario is a candidate for target-platform validation. Local CUDA tests
do **not** establish that 48 events per MI250X device fit across the full occupancy
distribution, or validate ROCm FlexAttention replay or eight-rank DDP.

## Local measurements, 2026-10-14

RTX 5060 Ti (16 GiB), PyTorch 2.11.0+cu128, BF16, full recommended hit model.
Five real `cld_edm_ttbar_hits/1:3.2.1` training batches per configuration, followed
by validation on separate test-split events. Each configuration ran in a fresh
process. Inputs were neither truncated nor synthesized. The same batch size uses
the same event indices and initialization seed in both modes.

| Events/GPU | Checkpointing | Peak allocated, MiB | Peak reserved, MiB | Mean step, ms |
|---:|:---:|---:|---:|---:|
| 1 | off | 3,429 | 3,778 | 196 |
| 1 | on | 887 | 1,132 | 233 |
| 4 | off | 13,960 | 15,470 | 605 |
| 4 | on | 3,171 | 4,124 | 761 |
| 16 | on | 12,610 | 15,046 | 3,266 |

Memory is the maximum forward/loss/backward peak across all five steps, measured
with `reset_peak_memory_stats`, `max_memory_allocated` and `max_memory_reserved`;
it is not step-end occupancy. Reserved memory includes the allocator cache.
Step time includes the real Hungarian matching, auxiliary/grouping losses,
backward, LAMB and scheduler update. It excludes input loading, host-side gradient
checks and snapshot transfers, and omits the first two steps for compilation and
warmup. These are short smoke measurements, not a throughput or convergence study.

At batch four, allocated memory falls **77%**, with **26%** more step time.
Batch sixteen completes training and validation on this card, with up to 13,312
padded hits/event in the measured batches. This demonstrates a fourfold increase
over the tested uncheckpointed batch four, which was already near capacity; it
does not establish a safe maximum for every event in the dataset.

Raw reports are in `experiments/heptv2_checkpointing_local/`: `baseline_b1.json`,
`checkpointed_b1.json`, `baseline_b4.json`, `checkpointed_b4.json`, and
`checkpointed_b16.json`. `scripts/studies/benchmark_heptv2_checkpointing.py`
reproduces the measurements and records event indices, lengths and loss components.

## Correctness and integration

- CPU/GPU tests compare all primary, auxiliary and grouping outputs and parameter
  gradients, with dropout, unequal event lengths, padding and a frozen encoder.
  They also check RNG preservation, strict weight loading and bypass in evaluation
  or `no_grad`. CPU regression suite: 94 passed, 2 GPU tests skipped. GPU HEPTv2
  and checkpoint suite: 14 passed.
- Full-model real-data parity with deterministic PyTorch algorithms enabled:
  **all 19 output tensors and the loss match exactly**. Across 229 parameter
  gradients, checkpointing changes the relative L2 norm by **0.132%**, compared
  with **0.129%** for an unchanged baseline repeat. Maximum absolute difference
  is **0.125** for both. Fused BF16 backward still has numerical variation.
- With normal nondeterministic execution, small forward differences and Hungarian
  assignment changes can amplify across optimizer updates; the five-step loss
  trajectories are not bitwise-identical. `parity_b1.json` and
  `parity_deterministic_b1.json` record repeated passes without optimizer updates
  to separate replay parity from this effect. Deterministic mode is diagnostic
  only and is not enabled in the training scenario.
- The existing scenario launcher completed four batch-eight training steps,
  validation and prediction at steps two/four, saved model/optimizer checkpoints,
  and resumed successfully from step two through step four. Artifacts are under
  `experiments/heptv2_checkpointing_local/pipeline/` and `resume/`.
- Both H100 and LUMI profiles resolve and pass scenario invariant checks. The
  local-spec helper now includes the recommended HEPTv2 model, so local smoke
  runs correctly restrict it to the available ttbar dataset configuration.

## Reproduce and launch

Run each memory measurement in a fresh process, without competing GPU workloads:

```bash
uv run python3 scripts/studies/benchmark_heptv2_checkpointing.py \
  --batch-size 4 --output experiments/heptv2_checkpointing_local/baseline_b4.json
uv run python3 scripts/studies/benchmark_heptv2_checkpointing.py \
  --batch-size 4 --checkpointing \
  --output experiments/heptv2_checkpointing_local/checkpointed_b4.json
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python3 \
  scripts/studies/benchmark_heptv2_checkpointing.py \
  --batch-size 1 --parity-only --deterministic \
  --output experiments/heptv2_checkpointing_local/parity_deterministic_b1.json
```

Local pipeline smoke test (the local wrapper deliberately overrides the production
budget; these are **not** batch-384 production settings):

```bash
GLOBAL_BATCH_SIZE=8 NUM_STEPS=4 VAL_FREQ=2 CHECKPOINT_FREQ=2 \
NVALID=8 NTEST=8 NUM_WORKERS=0 VALIDATION_DIAGNOSTICS_BATCHES=0 \
  bash scripts/local/train_scenario.sh cld_pf_heptv2_batch512_50k \
  --variant hit_heptv2_grouped_tracker_features
```

On LUMI, first repeat the benchmark in the existing LAIF environment with
`--platform configs/training/platforms/lumi_mi250x.yaml`, the LUMI TFDS directory
as `--data-dir`, and `--batch-size 48 --checkpointing`. Also run a short eight-rank
training/validation test at the target batch. Record allocated/reserved peaks and
step times, including high-occupancy batches, before committing a full allocation.

Inspect the full recipe without submitting:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cld_pf_heptv2_batch512_50k.yaml \
  --platform configs/training/platforms/lumi_mi250x.yaml --dry-run
```

After target-platform validation, the existing submission wrapper is:

```bash
bash scripts/lumi/train_scenario.sh cld_pf_heptv2_batch512_50k mi250x
```

This submits both comparison variants. No production job was submitted during
the local implementation and tests.
