# Reusable training scenarios

To repeat the single-device GH200 batch sweep on **LUMI MI250X or Flatiron H100**,
follow the [cross-platform reproduction runbook](../../docs/training/hpc-batch-sweep.md).
It includes branch checkout, data verification, site environment setup,
single-device launchers, qualification and full-sweep commands.

For the EmpireAI ARM64 GH200 test node and `ghtest` reservation, see the
[setup and smoke-test instructions](../../scripts/empireai/README.md).
`cld_gh200_backbone_benchmark` compares attention and HEPTv2 on the copied ttbar
hits subset at batch 8 for 1,000 steps with matched decoder, features and seed.

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
  --scenario configs/training/scenarios/cld_hits_output_comparison.yaml \
  --platform configs/training/platforms/local.yaml \
  --global-batch-size 8 \
  --dry-run
```

## CLIC and CLD backbone benchmark

[`clic_cld_backbone_benchmark`](scenarios/clic_cld_backbone_benchmark.yaml) runs
eight models per seed: CLIC/CLD × tracks-and-clusters/raw-hits × attention/HEPTv2.
Variant names are `{cld,clic}_{pf,hits}_{attention,heptv2}`. The older
`clic_cld_pf_set_hits_comparison` contains only attention models and retains its
historical recipe.

Track/cluster variants request **one GPU per job**; hit variants use the platform
GPU count (eight on H100 and MI250X). With global batch 128, this gives 128 events
on the single PF GPU and 16 events per GPU for eight-GPU hit jobs. Submission
automatically creates separate arrays: indices `0,1,4,5` request one GPU, while
`2,3,6,7` use the platform count. GPU counts, batch resolution, and worker allocation
checks follow each variant. Other Slurm resources still come from the platform;
in particular, LUMI's 480G host-memory request remains unchanged.

All eight models train from scratch for 50,000 steps at global batch 128 with
LAMB, LR 0.001 and cosine decay (6.4M event presentations each). Validation and
checkpoints run every 1,000 steps, matching the completed readout study. Seed 12345 is
the initial benchmark; repeat with `--seed` for training-seed uncertainty.
Both backbones have six layers and embedding width 256, with 16 heads.
HEPTv2 uses block size 128, deterministic hashing and activation checkpointing.
This matches event exposure and embedding width, not parameter count or compute.

The four hit models share the small-model recipe with the tracker momentum head
from the completed readout campaign in
[`20261021_cld_tracker_losses_readout`](../../notebooks/studies/20261021_cld_tracker_losses_readout/cld_tracker_losses_readout.ipynb): four global Flash
decoder layers, 256 slots, grid-diverse aggregate proposals, supervised hard
hit grouping, and geometry plus calorimeter/tracker-neighborhood features.
The decoder and input features match `cld_pf_heptv2_default_50k` with
`tracker_momentum_readout: true` for both backbones. In the matched 50k campaign,
the head improved charged width from 12.78% to 12.40%, tails from 3.69% to 3.38%,
and jet IQR from 0.10316 to 0.09815. Its +0.15 percentage-point F1 change has a
paired 95% interval spanning zero; the tail reduction interval excludes zero.
This is the best-supported current benchmark candidate, not a multi-seed proven
replacement for the conservative default in the [evidence summary](../../docs/cld_hit_results.md).
Sparse pooling, momentum/doublet inputs, fitted readouts, balanced CE, and auxiliary
fit/overlap losses stay off. The loss campaign only has matched 32k results;
combining balanced CE with the head is untested. The scaling study favors keeping
width 256 and depths 6/4 for particle reconstruction, although large improves jets.
CLIC and attention transfer of this recipe are new benchmark measurements, not previously
validated wins. Track/cluster models retain elementwise output.

The spec currently selects PF TFDS **3.3.0** and hits **3.2.2** for both detectors.
These are fresh comparisons, not reproductions of archived CLD **3.2.1** results.
Use common evaluation definitions and matched checkpoints, and compare momentum
resolution/tails, efficiency, jets and event closure alongside particle F1.

Preview all eight jobs without launching training:

```bash
uv run --no-sync python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/clic_cld_backbone_benchmark.yaml \
  --platform configs/training/platforms/flatiron_h100.yaml --dry-run
```

Submit with `scripts/flatiron/train_scenario.sh clic_cld_backbone_benchmark h100`.
Use `--variant clic_hits_heptv2` to select a single model. The local and Flatiron
profiles map data paths separately for CLIC and CLD. The existing LUMI profile
has one CLD directory for all productions; before using it for the full benchmark,
provide a platform with `data_dir: {cld: /path/to/cld, clic: /path/to/clic}` or a
shared TFDS root containing both detectors. A dry run resolves configurations
but does not verify data availability or GPU memory usage.

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
scripts/local/train_scenario.sh cld_hits_output_comparison --dry-run
scripts/local/train_scenario.sh cld_hits_output_comparison --seed 2468
```

Additional arguments are forwarded to the generic scenario runner, such as
`--variant set`, `--global-batch-size 4`, or `--set num_steps=100`.

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
scripts/flatiron/train_scenario.sh cld_hits_output_comparison h100 --dry-run
scripts/flatiron/train_scenario.sh cld_hits_output_comparison h100
```

The picker reads Slurm resources from the selected platform profile and derives
the array size from the scenario's variants and seeds. Use `--seed N` to submit
one comparison pair with an explicit seed.

Tallinn and LUMI use the same interface with site-specific profiles and workers:

```bash
scripts/tallinn/train_scenario.sh --list
scripts/tallinn/train_scenario.sh cld_hits_output_comparison l40 --dry-run
scripts/tallinn/train_scenario.sh cld_hits_output_comparison l40

scripts/lumi/train_scenario.sh --list
scripts/lumi/train_scenario.sh cld_hits_output_comparison mi250x --dry-run
scripts/lumi/train_scenario.sh cld_hits_output_comparison mi250x
```

For the three CLD hit-set references (Flash backbone with global Flash decoder,
HEPTv2 backbone with global Flash decoder, and HEPTv2 backbone with sectorized
Flash decoder), preview and submit the comparison campaign:

```bash
scripts/lumi/train_scenario.sh cld_set_hits_heptv2_sector_comparison mi250x --dry-run
scripts/lumi/train_scenario.sh cld_set_hits_heptv2_sector_comparison mi250x
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

`scenarios/pf_fm_experts.yaml` defines CLIC, CLD, IDEA and ColliderML PU0
experts with the same 256-wide, six-layer attention backbone, optimizer, learning
rate, global batch of 16 events and maximum update budget. Each expert uses one
GPU with `h100_single` and retains its detector's native input layout and train/validation samples.
IDEA uses the locally available 0.2.1 data.
All four experts and the joint run enable input normalization using the
statistics for their exact dataset versions. Historical expert checkpoints
from before this setting was enabled used unnormalized inputs.
Scenario filenames, names, and experiment directories use `pf_fm_experts`,
`pf_fm_joint`, and `pf_fm_finetuning`.
GPU counts come from the platform, with no `pf_fm` variant GPU overrides:
`h100_single` selects one GPU and `h100` selects eight. A four-GPU profile selects
four (`a100` is the existing four-GPU Flatiron profile). Global batch remains
16 events in every case, divided across the selected GPUs. The examples below
use `h100_single`; choose `h100` for multi-GPU training.
The two H100 training profiles share dataset roots and runtime settings. Flatiron
training profiles use eight loader workers per GPU and own Slurm CPU/memory/time
requests; scenarios own the model, optimizer,
loss patience, validation cadence, dataset selection, and global batch.
`flatiron_h100_benchmark` preserves the short four-hour, four-worker benchmark
configuration and environment-provided dataset roots; use it for batch sweeps.

All use six shared attention layers, width 256 (16 heads × 16 channels), native
Flash attention with padding excluded, and zero feed-forward dropout. HEPTv2
configuration is cleared. These are fresh runs; archived HEPTv2 checkpoints
cannot be resumed or loaded into the attention backbone.

All experts run up to 100,000 optimizer updates, stopping after three consecutive
validation cycles without a strictly lower total validation loss (`patience: 3`).
Fine-tuning uses the same stopping rule; actual training exposure can differ.
Both the expert and joint scenarios enable per-worker background-thread
benchmark recording with one-second telemetry sampling, 100 warmup steps excluded
from steady-state throughput, and optional CUDA-event timing disabled. Per-rank
telemetry, step measurements and phase boundaries are saved under each run's
`benchmark/` directory. TensorBoard utilization summaries cover rank zero;
use the per-rank telemetry files to compare all GPUs.
Validation runs every 10,000 steps on up to 256 events per sample in the
expert, joint, and fine-tuning campaigns. Each validation records particle
counts, matching efficiency/purity, PID accuracy, matched kinematic errors, and
event energy, scalar/vector transverse momentum, and MET metrics under
`metrics/particle/*` in TensorBoard and the validation history. Checkpoints
remain every 2,000 steps. The lowest-loss validation checkpoint is retained as
`checkpoints/best.pth`, independently of periodic checkpoint cleanup. Training
diagnostics run every 1,000 steps. Joint training retains its fixed update budget.
All runs use LAMB with a 0.0005 peak learning rate and the same OneCycle schedule
(30% warmup). Runs use validation loss for selection. Every
10,000 steps they also run test inference and plotting with `ntest: 256` per
sample (250 events for samples with ten TFDS configurations, due to equal
per-configuration allocation). Jet response median/IQR, matching counts,
recall, precision, F1, and fake rate are recorded in TensorBoard and the step
history under `step/<sample>/jet_ratio/jet_ratio_target_to_pred_pt/*`. Nominal
experts and the joint run use their recipe test samples; each CLD scan evaluates
only its available Z→ττ (ECAL) or ttbar (HCAL) sample at that variation’s data
root. Expert/fine-tuning pairs use identical test selections. Per-cycle plots
and predictions are retained in each experiment directory.

Preview the commands without submitting Slurm jobs:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/pf_fm_experts.yaml \
  --platform configs/training/platforms/flatiron_h100_single.yaml --dry-run
```

The full joint configuration is `scenarios/pf_fm_joint.yaml`. It uses the
`pyg-cldcliccolliderml-v1` recipe from main, including its per-source TFDS roots,
and enables `model.feature_bag` with input normalization. It runs 100,000 updates
on one GPU, with 16 events per step on CLIC, CLD, and ColliderML PU0 samples. Its attention
backbone, heads, optimizer, peak learning rate, and OneCycle schedule match the
experts. Only the input layer size varies with detector feature width and the
joint feature bag.

## Joint attention fine-tuning

[`pf_fm_finetuning`](scenarios/pf_fm_finetuning.yaml) contains IDEA and
the 12 [CLD granularity variants](#cld-granularity-experts-and-fine-tuning).
Further detector-configuration variants can share its joint checkpoint and
training settings through their own model/production and dataset overrides.

IDEA fine-tuning uses the expert's IDEA 0.2.1 qq, ttbar and fully hadronic WW
samples, shard 1, seed 12345, global batch 16 on one GPU, up to 100,000 **new** updates,
LAMB and the same 0.0005 OneCycle schedule. Validation/checkpoint intervals,
sample caps, padding, sampler mode and background benchmarking match the expert.
All model parameters are trainable. The joint checkpoint's feature bag and input
normalization remain enabled to keep learned tensor shapes compatible. The
feature bag differs from the expert, so the comparison includes the input
representation as well as pretraining. Input normalization is enabled in both
current scenarios. IDEA remains an engineering pipeline dataset with
truth-seeded proxy tracks, as documented in its recipe.

`load_weights_only: true` starts a fresh optimizer, schedule, RNG stream, sampler
and step counter. `reset_task_loss_weights: true` discards the joint task-loss
calibration and calibrates anew on IDEA for 100 steps. Strict model-weight
loading remains enabled. Independent calibration means native weighted loss
totals are not a common objective across the expert and fine-tuned models;
compare raw components or evaluate with common weights and physics metrics.

Train the attention joint model before starting these fine-tunes. The
`load` entry is an explicit placeholder under
`experiments/pf_fm_joint/REPLACE_WITH_JOINT_RUN/checkpoints/checkpoint-100000.pth`.
Use `--set load=/path/to/the/attention/joint/checkpoint-100000.pth` with the direct
runner, or edit `load` in the scenario before submitting through a site wrapper.
Strict loading rejects the old HEPTv2 checkpoint. The source checkpoint's GPU
count does not constrain this one-GPU fine-tune.

IDEA 0.2.1 normalization statistics are provided in
`mlpf/model/input_normalization_stats/idea_0.2.1.yaml`, computed from all 24,000
training events (8,000 each of qq, ttbar and fully hadronic WW). Test events
are excluded. IDEA 0.1.0 statistics cannot substitute for 0.2.1 because cluster
features changed. To regenerate the statistics on a host holding the dataset,
for example using the IDEA directory in the Flatiron profile:

```bash
export IDEA_TFDS_DIR=/mnt/ceph/users/jpata/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/tfds
uv run python3 scripts/compute_input_normalization_stats.py map idea 0.2.1 \
  --data-dir "$IDEA_TFDS_DIR" --work-dir results/input_normalization/idea_0.2.1 \
  --samples edm_qq edm_ttbar edm_ww_fullhad --shard 0 --num-shards 1
uv run python3 scripts/compute_input_normalization_stats.py reduce idea 0.2.1 \
  --work-dir results/input_normalization/idea_0.2.1
```

This writes `mlpf/model/input_normalization_stats/idea_0.2.1.yaml`. Keep that file
with the run's provenance and make it available on the submission/training host.
Then preview the fine-tune:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/pf_fm_finetuning.yaml \
  --platform configs/training/platforms/flatiron_h100_single.yaml --variant idea \
  --set load=/path/to/the/joint/checkpoint-100000.pth --dry-run
```

For site submission, set the checkpoint path in the scenario (or a copied
scenario), then use, for example,
`scripts/flatiron/train_scenario.sh pf_fm_finetuning h100_single --variant idea`.
The direct runner accepts `--set` overrides; the site submitter does not.
The scenario runner's `--continue` resumes a compatible fine-tune from its
own checkpoint, retaining optimizer, schedule, sampler, step and calibration
rather than restarting weights-only fine-tuning. A new experiment records its
source path and reset settings in the scenario manifest. No jobs are submitted
by adding this scenario.

## CLD granularity experts and fine-tuning

The expert and fine-tuning scenarios each contain these same-named variants,
giving 12 independent expert/fine-tune pairs. The inventory comes from the
local `*_pf/*/3.3.0/` datasets, excluding the neighboring hit datasets.

| Variant | Scan subdirectory | PF sample | Train events |
| --- | --- | --- | ---: |
| `cld_ecal_1p5mm` | `ecal_1p5mm` | `cld_edm_z_tautau_pf` | 8,996 |
| `cld_ecal_2mm` | `ecal_2mm` | `cld_edm_z_tautau_pf` | 8,996 |
| `cld_ecal_2p5mm` | `ecal_2p5mm` | `cld_edm_z_tautau_pf` | 8,997 |
| `cld_ecal_3mm` | `ecal_3mm` | `cld_edm_z_tautau_pf` | 8,994 |
| `cld_ecal_4mm` | `ecal_4mm` | `cld_edm_z_tautau_pf` | 8,995 |
| `cld_ecal_5p1mm` | `ecal_5p1mm` | `cld_edm_z_tautau_pf` | 8,997 |
| `cld_hcal_10mm` | `hcal_10mm` | `cld_edm_ttbar_pf` | 9,000 |
| `cld_hcal_12mm` | `hcal_12mm` | `cld_edm_ttbar_pf` | 9,000 |
| `cld_hcal_15mm` | `hcal_15mm` | `cld_edm_ttbar_pf` | 9,000 |
| `cld_hcal_20mm` | `hcal_20mm` | `cld_edm_ttbar_pf` | 9,000 |
| `cld_hcal_25mm` | `hcal_25mm` | `cld_edm_ttbar_pf` | 9,000 |
| `cld_hcal_30mm` | `hcal_30mm` | `cld_edm_ttbar_pf` | 9,000 |

Each dataset has ten TFDS config IDs (`1`–`10`) and 1,000 test events in total.
The train and validation configurations select only the process actually
available for that geometry; they replace the nominal qq/ttbar/WW sample map
rather than adding to it. The existing four nominal experts and IDEA fine-tune
remain available. Running the entire expert scenario now selects 16 jobs and
the entire fine-tuning scenario selects 13, per seed. Use `--variant` to select
just one geometry.

Both members of each pair use six-layer, width-256 attention with the feature bag,
input normalization, native Flash attention, and all
parameters trainable. In particular, these scan experts enable the feature bag
to match the joint checkpoint's input architecture, unlike the nominal experts.
They use the same fixed `cld_3.3.0.yaml` normalization tables, measured on the
nominal CLD training data. These are shared reference statistics, not estimates
recomputed from each scan or its test events.

Apart from the checkpoint-loading/reset fields, expert and fine-tuning resolved
configurations are identical within each geometry. The expert starts from
scratch, while the fine-tune strictly loads the common joint checkpoint with a
fresh optimizer, schedule, sampler, step counter and task-loss calibration.
Both use one GPU, global batch 16, seed 12345, up to 100k updates, LAMB, the 0.0005
OneCycle schedule, 10k validation intervals, 2k checkpoints, and 1-second background
telemetry. Both stop after three consecutive validation cycles without loss
improvement. The approximately 9k training events imply about 178 dataset passes
at the maximum budget.

Each variant overrides the nominal production's TFDS directory with
the platform's `data_dir.cld_granularity` plus its scenario `data_subdir`
(for example, `ecal_3mm`). Set this platform root to the scan location on the
training host; the existing `/mnt/work/.../granularity_scan_2025-05-29` location
is retained in the Flatiron profiles until relocated.
The runner records this root in the resolved configuration and job manifest and
emits it as the actual `--data-dir` argument. Preview a matched pair:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/pf_fm_experts.yaml \
  --platform configs/training/platforms/flatiron_h100_single.yaml \
  --variant cld_ecal_3mm --dry-run
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/pf_fm_finetuning.yaml \
  --platform configs/training/platforms/flatiron_h100_single.yaml \
  --variant cld_ecal_3mm --set load=/path/to/joint/checkpoint-100000.pth --dry-run
```

On another host, the direct runner accepts
`--set data_dir=/path/to/granularity_scan_2025-05-29/ecal_3mm` for the selected
variant. This takes precedence over the variant's root; `--data-dir` changes
the platform default, which the variant root otherwise overrides. For site
submission, adjust the root/checkpoint in the scenario or a copied scenario,
then use the submitter's repeatable `--variant` option to select geometries.
Experiments remain separate under `pf_fm_experts/cld_<geometry>_seed...`
and `pf_fm_finetuning/cld_<geometry>_seed...`.

## Portable benchmark sweeps

Use `scenarios/cld_backbone_benchmark.yaml` for new GPU comparisons and
`python -m scripts.training.submit_batch_sweep --platform <profile.yaml> --dry-run`
to preview allocation resources from a profile. Add `--python-wrapper` for a site
environment adapter. The historical GH200 and H100-3.2.1 scenarios retain their
original definitions. See [benchmark portability](../../docs/training/benchmark-portability.md)
for qualification, data provenance and the compact evidence release workflow.
