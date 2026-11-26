# Training scenarios

A scenario is a named, reproducible training campaign: one or more model variants, seeds, and shared training settings. Scenarios separate the scientific recipe from the machine it runs on:

| File | Contains |
|---|---|
| `configs/training/scenarios/<name>.yaml` | Variants (model recipe plus overrides), seeds, global batch size, steps, validation cadence, and the settings variants may differ in |
| `configs/training/platforms/<profile>.yaml` | GPUs per job, data roots per production, loader workers, Slurm resources, and container or environment details |

The runner checks that variants differ only in the declared settings, converts the global batch size into a per-GPU batch for the platform, and writes a `scenario-manifest.json` into each experiment so a run can be resumed with the same command.

Use a scenario when you want a comparison that someone else can repeat. For a single exploratory run, call `mlpf/pipeline.py train` directly as in [Train a model](train.md).

## Run a scenario

Preview the resolved training commands without starting anything:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/release_checkpoints.yaml \
  --platform configs/training/platforms/local.yaml \
  --variant cld_pf_attention \
  --dry-run
```

Useful options:

| Option | Effect |
|---|---|
| `--variant NAME` | Run one variant; omit to run all |
| `--seed N` | Replace the scenario's seed list |
| `--global-batch-size N` or `GROUP=N,...` | Override the global batch, per train group if needed |
| `--data-dir PATH` or `PRODUCTION=PATH` | Override the platform's data roots |
| `--set KEY=VALUE` | Override one training or model field, e.g. `--set num_steps=200` |
| `--continue` | Resume an unfinished run of this scenario from its latest checkpoint |

A dry run resolves configurations but does not check that the data exist or fit in GPU memory.

## Submit to a cluster

Site wrappers turn a scenario into Slurm array jobs:

```bash
scripts/flatiron/train_scenario.sh release_checkpoints h100 --dry-run
scripts/lumi/train_scenario.sh release_checkpoints mi250x --dry-run
scripts/flatiron/train_scenario.sh --list
```

Slurm wrappers exist for `flatiron`, `lumi`, `tallinn`, and `empireai`; the second argument names an accelerator profile for that site. `scripts/local/train_scenario.sh SCENARIO` runs on the local machine without Slurm and applies short-run overrides. Submission options:

- `--variant NAME` (repeatable) submits a subset. Array task indices stay those of the full scenario, so a later `--continue` or a resubmission still maps to the same experiment.
- `--continue` resubmits only runs whose latest checkpoint is below `num_steps`.
- `--seed N` replaces the seed list.

Variants that need different GPU counts (for example single-GPU track/cluster models and full-node hit models) are split into separate arrays automatically. On LUMI, profiles with `slurm.partial_node` send jobs that use fewer GPUs than a node to the `small-g` partition.

### Restart one LUMI component

On LUMI, the wrapper also writes one Slurm script per variant and seed under `logs_slurm/submissions/<scenario>_<timestamp>/`. To restart only one failed component:

```bash
sbatch logs_slurm/submissions/<scenario>_<timestamp>/heptv2_seed12345.sh
```

Each launch reads the run's saved `scenario-manifest.json`, resumes from the highest-numbered checkpoint, and exits immediately if `num_steps` is already reached. A run lock prevents two jobs from training in the same directory. The scripts keep the container, overlay, and spec copy from submission time; source code is still read from the working checkout. Keep the submission directory and experiment directory until the run is complete.

## Maintained scenarios

The scenario directory also holds historical and single-study recipes. These are the ones to start from:

| Scenario | Purpose |
|---|---|
| `release_checkpoints` | Reference checkpoints for each release: CLD, CLIC, ColliderML PU0 (tracks/clusters and hits) and CMS, attention and HEPTv2, 100k OneCycle steps. Train each release from its tag with the scenario unchanged. |
| `clic_cld_backbone_benchmark` | CLIC/CLD × tracks-and-clusters/hits × attention/HEPTv2 at matched event exposure |
| `cms_backbone_comparison` | CMS attention vs HEPTv2 on TFDS 2.8.0 for 320k steps ([rationale](../studies/cms_training_exposure.md)) |
| `cms_input_normalization`, `colliderml_input_normalization` | Same model with and without [input normalization](train.md#input-normalization) |
| `pf_fm_experts`, `pf_fm_joint`, `pf_fm_finetuning` | Attention experts for the Key4hep v1.4.0 CLD, CLIC, ILD and IDEA datasets, ColliderML pretraining with four training-data fractions, and fine-tuning of the pretrained model on each Key4hep detector; the earlier three-detector campaign is under `scenarios/legacy/` |
| `cld_clic_colliderml_feature_bag` | Joint training with and without the shared [feature bag](train.md#train-on-several-detectors) |
| `cld_pf_heptv2_default_50k` | Training recipe for the recommended CLD hit-set model `pyg-cld-hits-heptv2-set-v1` |
| `cld_backbone_benchmark` | Fixed recipe for [portable GPU benchmarks](benchmark-portability.md) |

The [scenario README](https://github.com/jpata/particleflow/blob/main/configs/training/README.md) lists every scenario with its recipe, data versions, and platform requirements, including the study scenarios whose results are in the [research notes](../studies/index.md).

## Release checkpoints

`release_checkpoints` is trained from each release tag without changes, so that checkpoints of consecutive releases differ only in code. It covers CLD, CLIC, and ColliderML PU0 (tracks/clusters and hits) and CMS, each with attention and HEPTv2, for 100,000 OneCycle steps without early stopping. Track/cluster models use input normalization.

v3.3.0 is the first release with this recipe. Relative to the v3.2.0 checkpoints (CLD/CLIC 50k steps, CMS 100k, cosine decay), the schedule, the CLD/CLIC step count, the [CMS batches](../studies/cms_training_exposure.md), and input normalization changed, and ColliderML is new, so a v3.2.0 → v3.3.0 difference mixes code and recipe changes. Re-evaluate older checkpoints with the current evaluation code on the same test events before comparing physics metrics.

Before a full release run, smoke-test variants that have not been trained with the current recipe:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/release_checkpoints.yaml \
  --platform configs/training/platforms/lumi_mi250x.yaml \
  --variant colliderml_hits_heptv2 --set num_steps=200 --set val_freq=100 \
  --experiments-dir /tmp/release_smoke
```

## Benchmark telemetry

Per-worker benchmark recording (throughput, latency, device telemetry, memory) is off by default. Enable it with `--set benchmark.enabled=true`; the release, `pf_fm_*`, and batch-sweep scenarios enable it themselves. Records are written under each experiment's `benchmark/` directory. See [Portable training benchmarks](benchmark-portability.md#recorder-artifacts-and-metric-scope) for the artifact definitions.
