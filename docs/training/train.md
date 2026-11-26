# Train a model

This guide trains the standard CLD track-and-cluster model on configuration 1, version 3.4.0 of the `ttbar`, `WW`, and `qq` datasets. The [catalog](../datasets/catalog.md) lists compatible recipes and published versions. Start with the two-step CPU run to check the complete path. Use the GPU example only after that check succeeds.

A **training step** updates the model once using one dataset-group batch. A **validation cycle** pauses training to measure loss on a separate set of held-out events. A **checkpoint** is a saved model and training state from a particular step.

## Prerequisites

- Complete the [main installation](../getting-started/installation.md).
- [Download and verify](../datasets/download.md) configuration 1, version 3.4.0 of the three CLD training datasets. You can also produce matching data using [Key4HEP production](../datasets/key4hep.md).
- The commands below use the download guide's data root, `data/tfds/tensorflow_datasets/cld`. For locally produced data, replace it with the directory containing the three dataset-name directories (typically `<workspace>/tfds`). For a small integration check without downloading the full partition, use the [quickstart](../getting-started/quickstart.md).
- Run commands from the repository root.

The standard `pyg-cld-v1` recipe selects these three event types from configurations 1 through 10. `--data_config 1` restricts all three to the downloaded configuration.

The [configuration reference](../reference/configuration.md) explains recipe selection, command-line overrides, and how to inspect the resolved settings before training.

## Step 1: run a CPU software check

```bash
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-v1 \
  --production-name cld \
  --data-dir data/tfds/tensorflow_datasets/cld \
  --prefix docs-cld-cpu- \
  train \
  --data_config 1 \
  --gpus 0 \
  --dtype float32 \
  --model.attention.attention_type math \
  --gpu_batch_multiplier 1 \
  --num_workers 1 \
  --prefetch_factor 1 \
  --ntrain 10 \
  --nvalid 10 \
  --ntest 10 \
  --num_steps 2 \
  --val_freq 2 \
  --checkpoint_freq 2
```

This run checks the software integration. A useful trained model requires a full training and validation campaign. CPU time depends strongly on the machine. The run should end with a `VALIDATION` line and create an experiment directory beginning with `experiments/docs-cld-cpu-`.

The essential options are:

| Option | Meaning |
|---|---|
| `--model-name pyg-cld-v1` | Select the maintained CLD track-and-cluster architecture and dataset recipe. |
| `--production-name cld` | Select CLD paths and detector-specific settings. |
| `--data-dir` | Point to the directory directly containing TFDS dataset-name directories. |
| `--gpus 0` | Run on CPU. Use `1` for one GPU. |
| `--num_steps` | Stop after this many parameter updates. |
| `--val_freq`, `--checkpoint_freq` | Choose how often validation and checkpoint writing occur. |

All other values come from three layers, in order: typed defaults in `mlpf/conf.py`, the selected recipes in `particleflow_spec.yaml`, and command-line overrides. The program logs the final flattened configuration before it starts.

## Step 2: inspect the result

Find the newest matching directory:

```bash
ls -dt experiments/docs-cld-cpu-* | head -n 1
```

A successful two-step run contains at least:

```text
experiments/docs-cld-cpu-.../
├── checkpoints/checkpoint-02.pth
├── history/step_2.json
├── model_kwargs.pkl
├── hyperparameters.json
├── particleflow_spec.yaml
├── train-config.yaml
└── train.log
```

It may also contain TensorBoard data, predictions, and plots. See [Artifacts and directories](../reference/artifacts-and-directories.md) before moving or publishing a checkpoint.

Check that the history contains finite training and validation losses:

```bash
uv run python3 - <<'PY'
import json
from pathlib import Path

experiment = sorted(Path("experiments").glob("docs-cld-cpu-*"), key=lambda p: p.stat().st_mtime)[-1]
history = json.loads((experiment / "history" / "step_2.json").read_text())
print("train total:", history["train"]["Total"])
print("valid total:", history["valid"]["Total"])
PY
```

Finite numbers show that data loading, forward and backward passes, and validation completed. Model convergence requires a longer loss history and held-out performance checks.

## Step 3: start a GPU training

The following bounded engineering example reads configuration 1 and stops after 2,000 steps. Publication reproduction uses the archived recipe and artifacts associated with that publication.

```bash
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-v1 \
  --production-name cld \
  --data-dir data/tfds/tensorflow_datasets/cld \
  --prefix cld-ttbar-config1- \
  train \
  --data_config 1 \
  --gpus 1 \
  --dtype bfloat16 \
  --gpu_batch_multiplier 4 \
  --sampler_mode interleaved-shards \
  --num_workers 2 \
  --prefetch_factor 2 \
  --num_steps 2000 \
  --val_freq 1000 \
  --checkpoint_freq 1000
```

Memory use depends on event size, attention implementation, precision, and per-rank batch size. Lower `--gpu_batch_multiplier` first if the process runs out of GPU memory. Assess physics quality with the held-out checks in the [validation overview](../validation/overview.md).

## Other detectors

The command shape is the same; the production, model recipe, data root, and dataset version must agree:

| Detector | Production | Model recipe | Data version |
|---|---|---|---|
| CMS | `cms_run3` | `pyg-cms-v1` | 3.2.0 (scenarios use 2.8.0) |
| CLD | `cld` | `pyg-cld-v1` | 3.4.0 |
| CLIC | `clic` | `pyg-clic-v1` | 3.4.0 |
| ColliderML | `colliderml` | `pyg-colliderml-v1`, `pyg-colliderml-pu200-v1` | 1.5.0 |
| IDEA | `idea` | `pyg-idea-pipeline-v1` | 0.5.0 |
| ILD@FCC-ee | `ild` | `pyg-ild-v1` | 0.2.0 |
| MAIA | `maia` | `pyg-maia-v1` | 1.2.0 |

Hit inputs use the dedicated `*-hits-*` recipes and need substantially more memory. The [dataset catalog](../datasets/catalog.md) lists all recipes and which versions are published.

## Attention and padding

Native Flash attention automatically packs valid elements into a jagged representation so padding does not enter attention. It requires a compatible GPU and a head dimension divisible by 8, between 8 and 256. Use `--model.attention.attention_type math` for CPU runs; dense math attention applies a key-padding mask. `model.attention.use_flash_attn_varlen` optionally selects the external FlashAttention variable-length kernel and requires the matching package.

Keep the configured padding multiple for HEPTv2 and other block-based models. HEPTv2 uses deterministic integer hashing by default (`model.heptv2.deterministic_hashing: true`) and stable tie ordering. Its coordinates and bucket assignments can differ from older floating-point hashing, so validate older checkpoints before resuming or comparing predictions.

For sample ordering, weighted groups, and per-group losses, see [Dataset mixing](data-mixing.md). The [ONNX guide](../validation/onnx.md) describes the separate export contract and runtime profiles.

## Input normalization

Track and cluster features span very different ranges: energies from MeV to TeV, track parameters with occasional failed-fit outliers. With `model.input_normalization: true`, the model clips a few track parameters to physical bounds, applies `log1p` to positive, heavy-tailed quantities such as energies and hit counts, and standardizes every column with the mean and standard deviation of the dataset's training split. The release checkpoints and the `pf_fm_*` scenarios use it for all track/cluster models. Hit datasets do not support it.

```bash
... train --model.input_normalization true
```

The statistics are committed per dataset version in `mlpf/model/input_normalization_stats/<dataset>_<version>.yaml` (CMS 2.8.0, CLD and CLIC 3.3.0 and 3.4.0, ColliderML 1.5.0, IDEA 0.1.0–0.5.0, ILD 0.2.0). Training fails with a clear error if the file for a selected version is missing. Compute it for a new version from the full training split:

```bash
uv run python3 scripts/compute_input_normalization_stats.py map cld 3.4.0 \
  --data-dir /path/to/cld/tfds --work-dir results/input_normalization/cld_3.4.0
uv run python3 scripts/compute_input_normalization_stats.py reduce cld 3.4.0 \
  --work-dir results/input_normalization/cld_3.4.0
```

`map` can be split over Slurm tasks (`--shard`, `--num-shards` default to the `srun` task index and count). The statistics become part of the checkpoint, so inference and [ONNX export](../validation/onnx.md#models-with-input-normalization) do not need the YAML files. Normalized models need each event's dataset `source_id`, which the training and `test` pipelines supply automatically.

## Train on several detectors

`pyg-cldcliccolliderml-v1` trains one track/cluster model on CLD, CLIC, and ColliderML PU0, with one dataset group per detector and equal step weights. Select it with `--production-name cld_clic_colliderml`, which takes each detector's data root from its own production. To point at other roots, repeat `--data-dir` per detector: `--data-dir cld=/data/cld --data-dir clic=/data/clic --data-dir colliderml=/data/colliderml`. Scenario platform profiles map these roots by production name.

Each detector has its own feature layout. Two settings let one encoder handle all of them:

- `model.input_normalization: true` (required) brings the columns of each detector to a comparable scale.
- `model.feature_bag: true` maps each detector's columns into a shared vocabulary of named physical quantities, so that, for example, track pT always enters the encoder in the same slot. Quantities a detector does not provide are zero. It requires `model.input_encoding: split`.

The `cld_clic_colliderml_feature_bag` scenario compares joint training with and without the feature bag; `pf_fm_joint` and `pf_fm_finetuning` train a joint model and fine-tune it on IDEA or on CLD granularity variants. These are research workflows; see the [scenario guide](scenarios.md).

## CLD hit models

The recommended CLD hit research preset is `pyg-cld-hits-heptv2-set-v1`: a HEPTv2 backbone, a particle-set decoder with global attention, supervised hit grouping, and tracker-neighborhood features. Train it with the `cld_pf_heptv2_default_50k` [scenario](scenarios.md). The [CLD hit-model evidence](../studies/cld_hit_results.md) records the ablations behind this choice.

`pyg-cld-hits-heptv2-sector-set-v1` is an experimental variant in which decoder slots attend only to hits in nearby azimuthal sectors (`model.set_decoder.num_sectors`, default 32, and `sector_neighbors`, default 1). It avoids dense slot-by-hit attention but is not the recommended default.

HEPTv2 on GPU requires compiled `torch.nn.attention.flex_attention`; the explicit score calculation is used only on CPU. Pad hit counts to a multiple of the HEPTv2 block size. On an allocated GPU, check fused attention and checkpoint replay before a campaign:

```bash
python -m pytest -q tests/test_heptv2.py tests/test_heptv2_checkpointing.py
```

GPU cases skip when no GPU is visible. For training memory, see [HEPTv2 activation checkpointing](../studies/heptv2_checkpointing.md); on LUMI, `scripts/lumi/smoke_test.sh` checks the environment and RCCL.

## Next steps

- [Continue or fine-tune](resume-and-fine-tune.md) from a checkpoint.
- Run the `test` command and detector-level plots described in [Key4HEP validation](../validation/key4hep.md).
- Compare PyTorch and ONNX execution with [ONNX validation](../validation/onnx.md).
- Run a reproducible multi-variant campaign with a [training scenario](scenarios.md).
