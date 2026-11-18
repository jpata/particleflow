# Reusable training scenarios

To repeat the single-device GH200 batch sweep on **LUMI MI250X or Flatiron H100**,
follow the [cross-platform reproduction runbook](../../docs/training/hpc-batch-sweep.md).
It includes branch checkout, data verification, site environment setup,
single-device launchers, qualification and full-sweep commands.

For the EmpireAI ARM64 GH200 test node and `ghtest` reservation, see the
[setup and smoke-test instructions](../../scripts/empireai/README.md).
`cld_gh200_smoke` runs a short HEPTv2 hit-model smoke test there.

Native and Ray training workers collect portable benchmark evidence when enabled
with `--set benchmark.enabled=true`, under each experiment's `benchmark/` directory.
Recording is disabled by default; batch sweeps and the PF attention expert/joint
scenarios enable it explicitly. Use `python -m mlpf.benchmark_report
experiments/<scenario>` to export per-worker metrics and allocation work totals to
JSON/CSV. Configuration lives under `benchmark` (`enabled`, `sample_interval_seconds`,
`warmup_steps`, `gpu_timing_interval`); the scenario runner accepts these through
`--set`, and platform profiles can set them as runtime overrides. See the
[artifact and metric definitions](../../scripts/empireai/README.md#utilization-evidence-for-an-allocation-request).

The recommended CLD hit-model default is `pyg-cld-hits-heptv2-set-v1` in
`particleflow_spec.yaml`. Use `cld_pf_heptv2_default_50k` for its training recipe:

```bash
scripts/lumi/train_scenario.sh cld_pf_heptv2_default_50k mi250x --dry-run
scripts/local/train_scenario.sh cld_pf_heptv2_default_50k --dry-run
```

This promotes the 50k `hit_heptv2_grouped_tracker_features` result: six HEPTv2
layers, four global Flash decoder layers, 256 slots, supervised hit grouping,
and tracker-neighborhood features. Circle-fit readout, bounded fit correction,
detector-balanced grouping loss and event-consistency loss are disabled.
The recipe uses learning rate 0.001, global batch 64 and seed 12345; the local
launcher applies its usual short-run overrides and global batch 1.

On the study's 1,536 test events, charged pT residual half-width improved from
16.53% to 14.89% and charged matching efficiency from 77.34% to 78.40% versus
the previous grouped reference. This is the single-seed result evaluated on
2026-10-13. Historical sector and comparison presets remain unchanged.

For the latest archived outcomes, incomplete-run limits and historical dataset/metric
differences, see [the evidence summary](../../docs/cld_hit_results.md). The
[three-seed width-only study](../../docs/tracker_width_scaling.md) and
[grouping-loss factorial](../../docs/tracker_grouping_loss_experiments.md) are
separate from the older joint width/depth campaign below.

Scientific comparisons live under `scenarios/`; machine-dependent paths and runtime
tuning live under `platforms/`. Run a scenario locally with:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cld_pf_heptv2_hit_grouping_50k.yaml \
  --platform configs/training/platforms/local.yaml \
  --global-batch-size 8 \
  --dry-run
```

## Hit-model recipes

All hit-input models use HEPTv2 with the set output on the general `*_full` datasets
(`full_elements` selects tracker hits, calorimeter hits, tracks and clusters):
`pyg-cld-hits-heptv2-set-v1` (recommended CLD model), `pyg-cld-hits-heptv2-sector-set-v1`,
`pyg-clic-hits-heptv2-set-v1`, `pyg-idea-hits-heptv2-set-v1` and
`pyg-colliderml-hits-heptv2-set-v1`. The attention and elementwise hit recipes and the
scenarios that compared them were removed; their definitions remain in the git history and
their results in the dated study archives.

## CMS backbone comparison

`cms_backbone_comparison` trains CMS on TFDS version **2.8.0**, selecting either
`--variant attention` or `--variant heptv2`. It retains the PU/no-PU ttbar, QCD
and Z→ττ training/validation samples and the PU test samples. Both variants use
three backbone layers, embedding width 512, 16 heads, zero dropout, the same seed
and a global batch of 64;
HEPTv2 uses deterministic hashing and block size 100. The schedule is 100,000
steps at learning rate 0.0004. Dataset versions are pinned in the scenario.

The default CMS attention model also uses width 512 (16 heads × 32 channels).
The comparison disables task queries for both backbones and pins TFDS 2.8.0;
the default model specification still selects TFDS 3.2.0.

Supply the CMS TFDS directory explicitly when the platform profile only maps
CLD/CLIC data paths:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cms_backbone_comparison.yaml \
  --platform configs/training/platforms/local.yaml \
  --data-dir /path/to/cms/tfds \
  --variant attention \
  --dry-run
```

Use `--variant heptv2` for HEPTv2; omit `--variant` to run both. Remove
`--dry-run` to start training.

## CMS input normalization

`cms_input_normalization` compares `--variant norm` and `--variant no_norm`
using the HEPTv2 recipe from `cms_backbone_comparison`, including the CMS 2.8.0
sample pins, global batch of 64 and 100,000-step schedule. The variants differ
only in `model.input_normalization`. The normalized run uses the committed
full-training-split statistics in `mlpf/model/input_normalization_stats/cms_2.8.0.yaml`.

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cms_input_normalization.yaml \
  --platform configs/training/platforms/local.yaml \
  --data-dir /path/to/cms/tfds \
  --dry-run
```

Omit `--variant` to run both variants. Remove `--dry-run` to start training.

## HEPTv2 model-size scaling (50k)

`scenarios/cld_pf_heptv2_model_scaling_50k.yaml` scales width and depth together
while retaining the previous baseline's 154 input channels: geometry, tracker
neighborhoods and calorimeter neighborhoods. The new tracker momentum and doublet
features are explicitly disabled.

| Variant | Embedding / encoder width | Backbone layers | Decoder layers | Total parameters | Trainable parameters |
|---|---:|---:|---:|---:|---:|
| `hit_heptv2_small` | 256 | 6 | 4 | 12,732,407 | 12,709,139 |
| `hit_heptv2_medium` | 384 | 9 | 6 | 42,114,141 | 42,063,111 |
| `hit_heptv2_large` | 512 | 12 | 8 | 98,898,211 | 98,808,667 |

Parameter counts are from instantiated MLPF models, excluding any training-time
loss weighter. Total counts include frozen HEPTv2 parameters; study scaling plots
use trainable counts. Decoder embedding width follows the backbone embedding width;
HEPTv2's internal MLP width follows its unchanged `mlp_ratio=4`. Head counts stay
at 16 for the backbone and 8 for the decoder, and all variants retain 256 slots,
the same proposal scheme, supervised hit grouping and objective. The small model
matches the baseline architecture, with backbone activation checkpointing enabled
for all three sizes. This is a joint width/depth comparison, not an ablation that
separately identifies the contribution of each.

All runs use global batch **128** (16 per GPU on eight-GPU LUMI), seed 12345,
50,000 steps, LAMB, LR 0.001 and cosine decay. Validation and checkpoint intervals
are both **1,000 steps**. Validation/test sample sizes remain 512, with four
diagnostic batches and validation linear probes enabled. The batch increase from
the previous default's 64 gives 6.4M event presentations over the full schedule.

Preview the three resolved commands without submitting training:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cld_pf_heptv2_model_scaling_50k.yaml \
  --platform configs/training/platforms/lumi_mi250x.yaml --dry-run
```

Submit with `scripts/lumi/train_scenario.sh cld_pf_heptv2_model_scaling_50k mi250x`.
The local launcher uses global batch 1 and its usual short-run overrides,
including validation/checkpoint intervals of 200 unless overridden. To retain
the scenario's requested intervals locally, set `VAL_FREQ=1000 CHECKPOINT_FREQ=1000`.
All three models passed a finite CPU forward check on 128 synthetic hits. The
21 October archive includes matched 42k particle and resource measurements;
small remains the practical reference, while large improves jets but worsens
inclusive charged momentum. See [the evidence summary](../../docs/cld_hit_results.md).

The scenario declares a global batch size. The runner derives
`gpu_batch_multiplier` from the number of GPUs and the dataset batch size, and rejects
non-integral combinations. It also resolves every variant through `MLPFConfig` and
checks that variants differ only in the fields listed by
`allowed_variant_differences`.

The local picker discovers the same scenario files and applies the local platform
profile and short-run defaults:

```bash
scripts/local/train_scenario.sh --list
scripts/local/train_scenario.sh cld_pf_heptv2_hit_grouping_50k --dry-run
scripts/local/train_scenario.sh cld_pf_heptv2_hit_grouping_50k --seed 2468
```

Additional arguments are forwarded to the generic scenario runner, such as
`--variant hit_heptv2_baseline`, `--global-batch-size 4`, or `--set num_steps=100`.

With multiple variants or seeds, jobs are ordered by seed and then by variant. A
Slurm array can select one job using `--task-index $SLURM_ARRAY_TASK_ID`.
`--seed N` replaces the scenario seed list, including when a task index is used.
The site shell launchers expose this as the `SEED` environment variable. Without an
override, seeds come from the scenario file and are recorded in both the resolved
configuration and run manifest.

The Flatiron profiles request a fixed 24 CPUs per GPU
(`slurm.cpus_per_gpu: 24`): one GPU requests 24 CPUs, four GPUs request 96,
and eight GPUs request 192. The request scales with the job's effective GPU
count, including scenario variant overrides, independently of dataloader workers
or dataset groups. When set, `cpus_per_gpu` takes precedence over the fixed
`cpus_per_task` setting. Non-Flatiron profiles retain their explicit CPU requests.

List the available scenarios and accelerators, then submit using the Flatiron
picker:

```bash
scripts/flatiron/train_scenario.sh --list
scripts/flatiron/train_scenario.sh cld_pf_heptv2_hit_grouping_50k h100 --dry-run
scripts/flatiron/train_scenario.sh cld_pf_heptv2_hit_grouping_50k h100
```

The picker reads Slurm resources from the selected platform profile and derives
the array size from the scenario's variants and seeds. Use `--seed N` to submit
one comparison pair with an explicit seed.

Tallinn and LUMI use the same interface with site-specific profiles and workers:

```bash
scripts/tallinn/train_scenario.sh --list
scripts/tallinn/train_scenario.sh cld_pf_heptv2_hit_grouping_50k l40 --dry-run
scripts/tallinn/train_scenario.sh cld_pf_heptv2_hit_grouping_50k l40

scripts/lumi/train_scenario.sh --list
scripts/lumi/train_scenario.sh cld_pf_heptv2_hit_grouping_50k mi250x --dry-run
scripts/lumi/train_scenario.sh cld_pf_heptv2_hit_grouping_50k mi250x
```

The Tallinn worker runs the repository's `uv` environment directly. LUMI uses
Local-LAIF and an overlay created with `bash scripts/lumi/setup_env.sh`. Both
submitter and worker activate `LUMI_VENV` (default `particleflow-laif-env` in the
repository) inside the LAIF container (`IMG` overrides its path). The submitter
resolves configurations in the container, then invokes host `sbatch`;
`PYTHON_EXECUTABLE` overrides its container-side interpreter. Even LUMI
`--list`/`--dry-run` requires that environment. For a portable configuration-only
preview, use `scripts/training/run_scenario.py --dry-run` with a platform file.

Use repeated `--set KEY=VALUE` options only for explicit one-off overrides. Every
resolved run writes `scenario-manifest.json` containing the scenario, platform,
seed, final configuration, command, and git revision. Runs are grouped as
`<experiments_dir>/<scenario_name>/<variant>_seed<seed>_<timestamp>/`.

## Grouped HEPTv2 tracking follow-up (50k)

`scenarios/cld_pf_heptv2_grouped_tracking_50k.yaml` defines five runs, all with
seed 12345, global batch 64, learning rate 0.001, 50,000 steps, and reconstruction
diagnostics on four validation batches per rank. Every variant changes one
mechanism from the completed grouped HEPTv2 model:

| Variant suffix | Change from `hit_heptv2_grouped_baseline` |
|---|---|
| `tracker_features` | Enable existing tracker-neighborhood features |
| `balanced` | Average tracker and calorimeter grouping losses with equal detector weight |
| `tracker_readout` | Add predicted-group geometric summaries and reliable circle-fit momentum anchors |
| `event_consistency` | Add visible event momentum/energy consistency, weight 0.1 |

Detector balancing preserves the background weight of 0.1 within each detector,
normalizes over the local training batch, and averages only detector populations
present in that batch. It does not rebalance individual particles.

The tracker readout uses predicted groups only. It fits transverse circles with
three Huber-reweighted least-squares iterations, in double precision, and passes
nine summaries to a zero-initialized learned slot projection: hit and surface
counts, radial and longitudinal spans, transverse spatial scale, fitted log-pT,
relative fit residual, conditioning, and a validity flag. A valid fit requires
at least four hits, three distinct (system, side, layer) surfaces, 100 mm radial
span, a Gram-matrix eigenvalue ratio above 1e-7, a radial RMS residual below 2%
of that span, and fitted pT in (0.1, 10,000) GeV. The conversion assumes unit
charge magnitude and the CLD 2 T solenoid, set explicitly by
`tracker_group_magnetic_field`. It is a geometric prior rather than a full
material-aware track fit.

Reliable fits supply log-pT and a massless energy estimate from the initial
slot eta to the decoder's learned residual scale head. Existing calorimeter
seed anchors are retained. Unreliable groups keep their learned scales; their
geometric summaries can still inform the decoder. Validation reports
`metrics/set_slot/tracker_fit_fraction` (over all slots) and
`tracker_fits_per_event` on diagnostic batches. These count valid fits after
excluding slots with calorimeter anchors, but before the charged-PID gate; they
are not counts of final predictions using fitted momentum.

Event consistency uses the final slots' soft presence probabilities to sum
visible px, py, pz and energy. Momentum residuals are normalized by the target
scalar momentum sum; energy residuals by target total energy, both floored at
1 GeV. Smooth-L1 momentum and energy terms are added with the configured weight,
outside automatic task weighting. The target visible momentum need not vanish.
This is a weak auxiliary objective; compare jet widths, vector closure, class
momentum errors, efficiencies, and count bias rather than total training loss.

Resolve all production jobs without launching training:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cld_pf_heptv2_grouped_tracking_50k.yaml \
  --platform configs/training/platforms/lumi_mi250x.yaml --dry-run
```

Submit on LUMI using the same scenario through `scripts/lumi/train_scenario.sh`.
For local smoke runs, `scripts/local/train_scenario.sh
cld_pf_heptv2_grouped_tracking_50k` uses global batch 1 by default to accommodate
local GPU memory; it is not the production batch-64 comparison.

The grouped tracking scenario also enables `validation_linear_probe` with a
16-event cap per rank. It fits fresh raw/embedding calorimeter continuation
rankers on diagnostic events and evaluates on disjoint events. TensorBoard,
history JSON and `VALIDATION_PROBE` log lines report accuracy, coverage and
support. See `docs/calo_continuation_probe.md`. Local batch-one runs need at least
eight diagnostic batches for sufficient fit/evaluation events.

The `cld_pf_heptv2_fit_correction_50k` scenario compares ordinary grouped
regression, legacy tracker anchors and a bounded learned correction to fitted
charged-particle pT. See [charged fit correction](../../docs/charged_fit_correction.md)
for fit gates, fallback behavior and limitations. It does not start training
automatically.

The same diagnostics also report tracker and calorimeter instance ARI for raw
features and embeddings. Their default per-rank split is 8 fitting, 4 threshold
calibration and 4 evaluation events; TensorBoard tags end in `/raw/ari` and
`/embedding/ari`. Calibration and final evaluation are disjoint.

The tracker-fit loss ablation is `cld_pf_heptv2_fit_loss_50k`: baseline,
truth-circle compatibility loss, differentiable curvature loss, and both losses.
All retain the learned inference readout. See
[the experiment design](../../docs/tracker_fit_loss_experiments.md) for objectives,
local commands and acceptance criteria.

## Small-model tracker readout and sparse association

`cld_pf_heptv2_tracker_readout_50k` compares the unchanged small scaling baseline,
a tracker-specific momentum residual, sparse top-4 differentiable hit-to-slot
pooling, and both additions. It retains batch 128, 50k steps, and validation and
checkpoint intervals of 1,000. See [implementation and comparison details](../../docs/tracker_momentum_readout.md).

```bash
scripts/lumi/train_scenario.sh cld_pf_heptv2_tracker_readout_50k mi250x --dry-run
```

## Track/cluster attention experts

Use the PF attention scenarios to train detector experts, a shared model, or
fine-tunes of the shared model:

| Scenario | Training data | Initialization |
| --- | --- | --- |
| [`pf_fm_experts`](scenarios/pf_fm_experts.yaml) | Separate CLD, CLIC, IDEA 0.3.0 and ColliderML PU0 experts, plus CLD granularity variants | Fresh weights |
| [`pf_fm_joint`](scenarios/pf_fm_joint.yaml) | CLD, CLIC and ColliderML PU0, using `pyg-cldcliccolliderml-v1` | Fresh weights |
| [`pf_fm_finetuning`](scenarios/pf_fm_finetuning.yaml) | IDEA 0.3.0 and CLD granularity variants | Compatible joint attention checkpoint |

All use six shared attention layers of width 256 (16 heads × 16 channels),
native Flash attention, input normalization, LAMB with peak learning rate
0.0005, and a OneCycle schedule with 30% warmup. The maximum budget is 500,000
updates with a global batch of 16 events. Nominal experts use native detector
inputs; joint training, fine-tuning and CLD scan experts enable the feature bag.

GPU counts, dataset roots and Slurm resources come from the platform profile.
`h100_single` uses one GPU, `a100` four, and `h100` eight; the global batch stays
16. The H100 training profiles share runtime settings and use eight loader
workers per GPU. Use `flatiron_h100_benchmark` for four-hour batch sweeps with
four workers and environment-provided dataset roots.

| Scenario | Validation interval | Early stopping |
| --- | ---: | --- |
| Experts | 10,000 updates | 3 consecutive validation cycles without lower total loss |
| Joint | 10,000 updates | Full update budget |
| Fine-tuning | 1,000 updates | 30 consecutive validation cycles without lower total loss |

Validation and test inference use up to 256 events per sample. Validation
records particle matching, PID, kinematic and event-level metrics under
`metrics/particle/*`. Test inference writes predictions, plots and jet-response
metrics. Periodic checkpoints are saved every 2,000 updates; the lowest-loss
checkpoint is retained separately as `checkpoints/best.pth`.
Per-worker benchmark telemetry is saved under `benchmark/`, sampled every
second with 100 warmup steps excluded from steady-state throughput.

Preview expert commands without submitting jobs:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/pf_fm_experts.yaml \
  --platform configs/training/platforms/flatiron_h100_single.yaml --dry-run
```

## Joint attention fine-tuning

[`pf_fm_finetuning`](scenarios/pf_fm_finetuning.yaml) trains all model parameters
on IDEA or a CLD granularity variant. IDEA selects qq, ttbar and fully hadronic
WW samples at version 0.3.0, configuration 1. Its tracks are truth-seeded
proxies; see the [IDEA dataset description](../../docs/datasets/idea.md).

Set `load` to a compatible joint attention checkpoint in the scenario or pass
`--set load=/path/to/joint/checkpoint.pth` to the direct runner:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/pf_fm_finetuning.yaml \
  --platform configs/training/platforms/flatiron_h100_single.yaml --variant idea \
  --set load=/path/to/joint/checkpoint.pth --dry-run
```

`load_weights_only: true` starts a fresh optimizer, schedule, RNG stream,
sampler and step counter. `reset_task_loss_weights: true` recalibrates task-loss
weights on the fine-tuning data. Use `--continue` to resume the fine-tune's own
checkpoint with its training state and calibration intact. See
[resume and fine-tuning](../../docs/training/resume-and-fine-tune.md) for loading
options and compatibility requirements. Compare raw loss components or physics
metrics when runs use independently calibrated task-loss weights.

For site submission, set `load` in the scenario or a copied scenario, then run
`scripts/flatiron/train_scenario.sh pf_fm_finetuning h100_single --variant idea`.
The direct runner accepts `--set` overrides; the site submitter uses the scenario
settings.

Input normalization requires statistics for the selected dataset version.
IDEA 0.3.0 uses `mlpf/model/input_normalization_stats/idea_0.3.0.yaml`.
To regenerate statistics from the training splits on a host holding the data:

```bash
export IDEA_TFDS_DIR=/path/to/idea/tfds
uv run python3 scripts/compute_input_normalization_stats.py map idea 0.3.0 \
  --data-dir "$IDEA_TFDS_DIR" --work-dir results/input_normalization/idea_0.3.0 \
  --samples edm_qq edm_ttbar edm_ww_fullhad --shard 0 --num-shards 1
uv run python3 scripts/compute_input_normalization_stats.py reduce idea 0.3.0 \
  --work-dir results/input_normalization/idea_0.3.0
```

The generated YAML must be available on the submission and training hosts.

## CLD granularity experts and fine-tuning

The expert and fine-tuning scenarios provide 12 matched pairs using CLD
track/cluster datasets at version 3.3.0:

| Variant names | Scan subdirectories | Sample |
| --- | --- | --- |
| `cld_ecal_{1p5mm,2mm,2p5mm,3mm,4mm,5p1mm}` | `ecal_<size>/tfds` | `cld_edm_z_tautau_pf` |
| `cld_hcal_{10mm,12mm,15mm,20mm,25mm,30mm}` | `hcal_<size>/tfds` | `cld_edm_ttbar_pf` |

Each variant uses TFDS configurations 1–10 for its selected sample in training,
validation and testing. Expert and fine-tuning pairs share the architecture,
feature bag, optimizer, global batch and sample selection. Experts start from
fresh weights; fine-tunes load the joint checkpoint and reset training state
and task-loss calibration. Validation cadence and patience follow the table
above. All scan variants use the nominal `cld_3.3.0.yaml` normalization tables
as shared reference statistics.

Set the platform's `data_dir.cld_granularity` to the scan root. The runner
appends each variant's `data_subdir`, such as `ecal_3mm/tfds`, and passes the
resolved directory to training. Use `--variant` to select a geometry:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/pf_fm_experts.yaml \
  --platform configs/training/platforms/flatiron_h100_single.yaml \
  --variant cld_ecal_3mm --dry-run
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/pf_fm_finetuning.yaml \
  --platform configs/training/platforms/flatiron_h100_single.yaml \
  --variant cld_ecal_3mm --set load=/path/to/joint/checkpoint.pth --dry-run
```

The direct runner's `--set data_dir=/path/to/variation/tfds` overrides the variant
root. `--data-dir` sets the platform default; a variant-specific root takes
precedence over that default. For site submission, configure the platform root
and scenario checkpoint, and select geometries with repeatable `--variant`
options. Experiments are stored under `<scenario>/<variant>_seed...`.

## Portable benchmark sweeps

Use `scenarios/cld_hits_heptv2_benchmark.yaml` (HEPTv2 set hit model) for new GPU comparisons and
`python -m scripts.training.submit_batch_sweep --platform <profile.yaml> --dry-run`
to preview allocation resources from a profile. Add `--python-wrapper` for a site
environment adapter. The earlier attention/HEPTv2 benchmark scenarios (GH200,
H100 with TFDS 3.2.1) were removed together with the attention hit recipes. See [benchmark portability](../../docs/training/benchmark-portability.md)
for qualification, data provenance and the compact evidence release workflow.
