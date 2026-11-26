# Training scenarios

A scenario (`scenarios/<name>.yaml`) defines a reproducible training campaign: variants, seeds, global batch size, and shared training settings. A platform profile (`platforms/<profile>.yaml`) supplies GPUs, data roots, loader settings, and Slurm resources. The user guide is [Training scenarios](../../docs/training/scenarios.md); this file is the reference for the runner and the individual scenarios. Study design and results live in [`docs/studies/`](../../docs/studies/index.md).

## Running

Preview locally with the generic runner:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/release_checkpoints.yaml \
  --platform configs/training/platforms/local.yaml \
  --variant cld_pf_attention --dry-run
```

Site pickers discover the same scenarios and submit Slurm arrays. The second argument is an accelerator profile (`--list` shows them):

```bash
scripts/flatiron/train_scenario.sh SCENARIO h100 --dry-run   # a100, h100, h100_single, h200
scripts/lumi/train_scenario.sh SCENARIO mi250x               # mi250x, mi250x_single
scripts/tallinn/train_scenario.sh SCENARIO l40
scripts/empireai/train_scenario.sh SCENARIO gh200
scripts/local/train_scenario.sh SCENARIO                     # no Slurm, short-run overrides, global batch 1
```

Common options: `--variant NAME` (repeatable; array indices stay those of the full scenario), `--seed N` (replaces the seed list; `SEED` in the shell launchers), `--continue` (resubmit runs whose latest checkpoint is below `num_steps`), and, for the direct runner, `--global-batch-size`, `--data-dir`, and `--set KEY=VALUE`.

## How the runner resolves a scenario

- **Batch size.** The scenario declares a global batch per step; a variant can override it with `global_batch_size`, and `--global-batch-size` overrides both (one value, or `GROUP=N,...` per train group). The runner derives `gpu_batch_multiplier` from the GPU count and the dataset batch size and rejects non-integral combinations.
- **Controlled comparisons.** Every variant is resolved through `MLPFConfig`; variants may differ only in the fields listed under `allowed_variant_differences`.
- **Jobs.** Jobs are ordered by seed, then variant; `--task-index $SLURM_ARRAY_TASK_ID` selects one. Variants with different GPU counts are submitted as separate arrays.
- **Outputs.** Runs go to `<experiments_dir>/<scenario>/<variant>_seed<seed>_<timestamp>/` with a `scenario-manifest.json` holding the scenario, platform, seed, final configuration, command, and git revision. `--continue` and the LUMI per-component scripts resume from it.
- **Resources.** Flatiron profiles request `slurm.cpus_per_gpu: 24`, scaled by each job's GPU count. Profiles with `slurm.partial_node` (LUMI) send jobs using fewer GPUs than a node to a shared partition (`small-g`, 7 cores and 60 GB per GCD).
- **Data roots.** `data_dir` in a profile is a path or a map from production name to path; `--data-dir` overrides it. Variants can append a `data_subdir` (used for the CLD granularity scans).
- **LUMI environment.** Submitter and worker run inside the LAIF container (`IMG`) with the overlay `LUMI_VENV` created by `bash scripts/lumi/setup_env.sh`; even `--list` and `--dry-run` need it. For a configuration-only preview, use `run_scenario.py --dry-run`.
- **Benchmark telemetry.** Off unless `benchmark.enabled=true` (`--set` or a profile runtime override); written to each experiment's `benchmark/` directory. `python -m mlpf.benchmark_report experiments/<scenario>` exports JSON/CSV. See [benchmark portability](../../docs/training/benchmark-portability.md#recorder-artifacts-and-metric-scope).

## Maintained scenarios

### `release_checkpoints`

Reference checkpoints published with each release under `{cld,clic,colliderml}/{clusters,hits}/vX.Y.Z/` and `cms/vX.Y.Z/` on the [model hub](https://huggingface.co/jpata/particleflow). Train each release from its tag with this file unchanged; a change starts a new comparison series and belongs in the release notes.

| Variants | Model | Data | Global batch | GPUs |
|---|---|---|---:|---|
| `{cld,clic}_pf_{attention,heptv2}` | `pyg-{cld,clic}-v1` | PF 3.3.0 | 128 | 1 |
| `{cld,clic}_hits_{attention,heptv2}` | `pyg-{cld,clic}-hits-set-v1` | hits 3.2.2 | 128 | platform |
| `colliderml_pf_{attention,heptv2}` | `pyg-colliderml-v1` | PU0 PF 1.5.0 | 128 | 1 |
| `colliderml_hits_{attention,heptv2}` | `pyg-colliderml-hits-set-v1` | PU0 hits 1.3.0 | 128 | platform |
| `cms_{attention,heptv2}` | `pyg-cms-v1` | PU and no-PU 2.8.0 | 96 PU / 768 no-PU | platform |

100,000 steps, OneCycle (30% warm-up), no early stopping. CLD, CLIC and ColliderML follow the backbone benchmark below (LAMB, peak LR 0.001, six layers of width 256, validation on 512 events every 1,000 steps); ColliderML hit models use 512 slots. CMS keeps the `cms_backbone_comparison` model (three layers of width 512, LR 0.0004, validation every 10,000 steps) with the [96/768 batches](../../docs/studies/cms_training_exposure.md). Track/cluster models use input normalization. Checkpoints are written every 1,000 steps (CMS 10,000) and all are kept; hit models need several 48-hour allocations, so resubmit them with `--continue`.

```bash
scripts/lumi/train_scenario.sh release_checkpoints mi250x --dry-run
```

### `clic_cld_backbone_benchmark`

CLIC/CLD × tracks-and-clusters/hits × attention/HEPTv2 (`{cld,clic}_{pf,hits}_{attention,heptv2}`), from scratch, 50k steps at global batch 128, LAMB, LR 0.001, cosine decay. Both backbones have six layers of width 256 with 16 heads; HEPTv2 uses block size 128 and activation checkpointing. This matches event exposure and width, not parameter count. Track/cluster jobs use one GPU, hit jobs the platform count. The hit recipe is described in [CLD hit-model campaigns](../../docs/studies/cld_hit_campaigns.md#hit-recipe-in-the-backbone-benchmark). The older `clic_cld_pf_set_hits_comparison` keeps its historical attention-only recipe.

```bash
scripts/flatiron/train_scenario.sh clic_cld_backbone_benchmark h100 --dry-run
```

The LUMI profile maps one CLD directory for all productions; give it a `data_dir` map with CLD and CLIC roots before running the full benchmark.

### `cms_backbone_comparison` and `cms_input_normalization`

`cms_backbone_comparison` trains CMS attention or HEPTv2 (`--variant attention|heptv2`) on TFDS 2.8.0: three layers of width 512, 16 heads, global batches 96 (PU) / 768 (no-PU), 320,000 steps at LR 0.0004 with cosine decay, validation on 5,000 events per sample, task queries disabled. [CMS training exposure](../../docs/studies/cms_training_exposure.md) explains these choices. `cms_input_normalization` compares `norm` and `no_norm` with the HEPTv2 recipe at global batch 64 and 100k steps, using `mlpf/model/input_normalization_stats/cms_2.8.0.yaml`.

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cms_backbone_comparison.yaml \
  --platform configs/training/platforms/local.yaml \
  --data-dir /path/to/cms/tfds --variant attention --dry-run
```

### `colliderml_input_normalization` and `cld_clic_colliderml_feature_bag`

`colliderml_input_normalization` trains the ColliderML PU0 track/cluster model with and without input normalization. `cld_clic_colliderml_feature_bag` trains the joint `pyg-cldcliccolliderml-v1` model (production `cld_clic_colliderml`, input-normalized) with and without the feature bag, for 2,000 steps at global batch 1,024.

### `pf_fm_experts`, `pf_fm_joint`, `pf_fm_finetuning`

| Scenario | Train, validation and test datasets | Initialization |
|---|---|---|
| `pf_fm_experts` | One model per Key4hep v1.4.0 detector (CLD, CLIC, ILD, IDEA) | Fresh weights |
| `pf_fm_joint` | ColliderML Release 1 ttbar without pileup, TFDS 1.5.0, configurations 1–10, with four training-data fractions | Fresh weights |
| `pf_fm_finetuning` | One model per Key4hep v1.4.0 detector, with the same data as its expert | Checkpoint from the 25% `pf_fm_joint` variant |

Each [Key4hep v1.4.0 detector](../../docs/datasets/key4hep-50k.md) uses five track/cluster samples (ttbar, fully hadronic WW, qq, Z→qq and Z→tautau). CLD and CLIC use TFDS 3.4.0, configurations 1–10; ILD uses 0.2.0 and IDEA 0.5.0, both with configuration 1. The scenarios use the committed normalization tables for these versions and for [ColliderML 1.5.0](../../docs/datasets/colliderml.md).

All three use four shared attention layers of width 512 (16 heads × 32 channels), native Flash attention, input normalization, LAMB and OneCycle with peak LR 0.001 and 30% warm-up, and global batch 32. Experts use native detector inputs; pretraining and fine-tuning use the same feature bag so that weights load strictly.

| Scenario | Update budget | Warm-up | Validation interval |
|---|---:|---:|---:|
| Experts | 100,000 | 30,000 | 10,000 |
| Joint | 100,000 | 30,000 | 10,000 |
| Fine-tuning | 30,000 | 9,000 | 1,000 |

Patience covers the full schedule. Resuming an older run restores the learning-rate schedule saved in its checkpoint, so start new expert and joint runs for these schedules. Checkpoints are written every 2,000 updates, and the lowest-loss one is kept as `checkpoints/best.pth`. The Flatiron `h100_single` profile requests one H100, 24 CPUs, 128 GiB of host memory and 48 hours per run.

```bash
scripts/flatiron/train_scenario.sh pf_fm_experts h100_single --dry-run
scripts/flatiron/train_scenario.sh pf_fm_joint h100_single --dry-run
scripts/flatiron/train_scenario.sh pf_fm_finetuning h100_single --dry-run
```

`pf_fm_joint` runs `data_25pct`, `data_50pct`, `data_75pct` and `joint` (100%) with the same budget, validation and test events. Each subset takes its fraction from every TFDS configuration, rounded down to whole events with at least one event per nonempty configuration; a seeded permutation keeps the subsets nested. Pass `--variant joint` for the 100% variant alone.

`pf_fm_finetuning` loads `checkpoint-100000.pth` from the `data_25pct` variant, trains all parameters, loads weights only, and recalibrates the task-loss weights on each detector; compare raw loss components or physics metrics across runs. Set `training.parameters.load` in the scenario, or pass `--set load=PATH` to the direct runner, to use another checkpoint. See [resume and fine-tuning](../../docs/training/resume-and-fine-tune.md).

The previous three-detector campaign, including the CLD granularity variants, is kept under `scenarios/legacy/`. CLD granularity variants are `cld_ecal_{1p5mm,2mm,2p5mm,3mm,4mm,5p1mm}` (`cld_edm_z_tautau_pf`) and `cld_hcal_{10mm,12mm,15mm,20mm,25mm,30mm}` (`cld_edm_ttbar_pf`), all at 3.3.0 with configurations 1–10 and the nominal `cld_3.3.0` normalization statistics. Set the platform's `data_dir.cld_granularity` to the scan root; each variant appends its `data_subdir` (e.g. `ecal_3mm/tfds`).

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/legacy/pf_fm_finetuning.yaml \
  --platform configs/training/platforms/flatiron_h100_single.yaml \
  --variant cld_ecal_3mm --set load=/path/to/legacy/joint/checkpoint.pth --dry-run
```

Input normalization needs statistics for every selected dataset version; compute missing ones as described in [Train a model](../../docs/training/train.md#input-normalization).

### `cld_pf_heptv2_default_50k`

Training recipe for the recommended CLD hit model `pyg-cld-hits-heptv2-set-v1`: LR 0.001, global batch 64, seed 12345, 50k steps. See [CLD hit-model campaigns](../../docs/studies/cld_hit_campaigns.md).

```bash
scripts/lumi/train_scenario.sh cld_pf_heptv2_default_50k mi250x --dry-run
```

### Portable benchmarks

`cld_backbone_benchmark` is the fixed recipe for GPU comparisons. Preview a sweep with `python -m scripts.training.submit_batch_sweep --platform <profile.yaml> --dry-run` (add `--python-wrapper` for a site environment adapter). See [benchmark portability](../../docs/training/benchmark-portability.md) and the [site setup](../../docs/training/hpc-batch-sweep.md). `cld_gh200_backbone_benchmark`, `cld_gh200_smoke`, and `cld_h100_backbone_benchmark_321` reproduce the historical v3.2.0 runs; for the EmpireAI GH200 node see [`scripts/empireai/README.md`](../../scripts/empireai/README.md).

## Study scenarios

These scenarios define completed or historical studies. They stay unchanged so that results can be reproduced; design and outcomes are in the linked notes. Each file's header comment states what it compares.

| Scenario | Study |
|---|---|
| `cld_pf_heptv2_model_scaling_50k`, `cld_pf_heptv2_grouped_tracking_50k` | [CLD hit-model campaigns](../../docs/studies/cld_hit_campaigns.md) |
| `cld_pf_heptv2_tracker_readout_50k` | [Tracker momentum readout](../../docs/studies/tracker_momentum_readout.md) |
| `cld_pf_heptv2_tracker_momentum_50k` | [Tracker momentum features](../../docs/studies/tracker_momentum_features.md) |
| `cld_pf_heptv2_fit_loss_50k` | [Tracker-fit supervision](../../docs/studies/tracker_fit_loss_experiments.md) |
| `cld_pf_heptv2_fit_correction_50k` | [Charged fit correction](../../docs/studies/charged_fit_correction.md) |
| `cld_pf_heptv2_tracker_grouping_losses_50k` | [Tracker grouping losses](../../docs/studies/tracker_grouping_loss_experiments.md) |
| `cld_pf_heptv2_tracker_width_scaling_50k` | [Tracker width scaling](../../docs/studies/tracker_width_scaling.md) |
| `cld_pf_heptv2_batch512_50k`, `cld_pf_track_cluster_batch6144_{10k,50k}` | [HEPTv2 activation checkpointing](../../docs/studies/heptv2_checkpointing.md) and its large-batch track/cluster counterparts |
| `cld_pf_heptv2_hit_grouping_50k`, `cld_pf_heptv2_momentum_50k_comparison`, `cld_pf_hit_models_200k_comparison`, `cld_pf_hits_comparison` | Earlier hit-grouping and momentum comparisons; see [CLD hit-model evidence](../../docs/studies/cld_hit_results.md) |
| `cld_set_hits_{heptv2_sector,improvement,object_formation,query_matching}_comparison`, `cld_hits_{backbone,output}_comparison` | Earlier set-decoder and backbone comparisons |
| `clic_cld_pf_set_hits_comparison` | Original CLD/CLIC track/cluster vs hit-set comparison |
