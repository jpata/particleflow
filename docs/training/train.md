# Train a model

This guide trains the standard CLD track-and-cluster model on configuration 1 of the published `ttbar`, `WW`, and `qq` datasets downloaded in the [dataset guide](../datasets/download.md). Start with the two-step CPU run to check the complete path. Use the GPU example only after that check succeeds.

A **training step** updates the model once using one accumulated batch. A **validation cycle** pauses training to measure loss on a separate set of held-out events. A **checkpoint** is a saved model and training state from a particular step.

## Prerequisites

- Complete the [main installation](../getting-started/installation.md).
- Download and verify configuration 1, version 3.2.1 of the three CLD training datasets as described in [Download a dataset](../datasets/download.md).
- Run commands from the repository root.

The standard `pyg-cld-v1` recipe selects these three event types from configurations 1 through 10. `--data_config 1` restricts all three to the downloaded configuration.

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
  --num_workers 2 \
  --prefetch_factor 2 \
  --num_steps 2000 \
  --val_freq 1000 \
  --checkpoint_freq 1000
```

Memory use depends on event size, attention implementation, precision, and batch accumulation. Lower `--gpu_batch_multiplier` first if the process runs out of GPU memory. Assess physics quality with the held-out checks in the [validation overview](../validation/overview.md).

## Experimental CLD hit-set model

The `pyg-cld-hits-heptv2-sector-set-v1` recipe uses a HEPTv2 hit backbone and an input-conditioned particle-set decoder. Decoder slots are assigned to periodic azimuthal sectors from their current reference direction. Each slot cross-attends only to hits in its sector and `sector_neighbors` sectors on either side; slot self-attention is also sector-local. This avoids dense slot-by-hit masks and global slot self-attention. A sector with no nearby hits uses its nearest hit so attention is always defined.

The recipe starts with 32 sectors, one neighboring sector on each side, and 256 slots. These settings can be overridden with `model.set_decoder.num_sectors`, `model.set_decoder.sector_neighbors`, and `model.set_decoder.num_slots`. In this mode, `local_attention_radius` limits the reference update per decoder layer; it is not an exact geometric cross-attention cutoff. The model requires hit counts padded to a multiple of the HEPTv2 block size (128 in this recipe).

HEPTv2 GPU bucket attention requires compiled `torch.nn.attention.flex_attention` and never falls back to materializing attention scores. Bucket counts use a dynamic batch dimension so varying padded event lengths do not create a separate compiled kernel for every number of buckets. Failures retain the original exception and report the input shape, dtype, and PyTorch/ROCm versions; the explicit score calculation is used only on CPU.

Run the small MI250X diagnostic before a training campaign:

```bash
sbatch scripts/lumi/debug_heptv2.sh
```

It checks fused attention outputs and gradients (including log-normalizers used to weight hashes), runs six-layer BF16 forward/backward/optimizer steps with varying event sizes and batches, and reports peak GPU allocation in `logs_slurm/heptv2-debug-<jobid>.out`. This is a synthetic smoke test, not a full training memory estimate: matching, data loading, and distributed training are not exercised.

On MI250X with LAIF PyTorch 2.11 / ROCm 7.2 (2026-10-07, job 22601649), all 34 HEPTv2/set-prediction tests and 16 six-layer training smoke cases passed. Peak allocation was 14.0 GiB for one 100,000-hit event and 11.2 GiB for 64 events with 1,024 hits each. The original static bucket-as-head layout reproduced Dynamo’s eight-recompilation limit and silent unfused/dense execution on the ninth distinct bucket count.

## Train CMS or CLIC

The command shape is the same, but the production, model recipe, data root, dataset version, and locally available samples must agree:

| Detector | Production | Model recipe | Typical data root |
|---|---|---|---|
| CMS | `cms_run3` | `pyg-cms-v1` | the CMS production workspace `tfds/` directory |
| CLD | `cld` | `pyg-cld-v1` | `.../tensorflow_datasets/cld` |
| CLIC | `clic` | `pyg-clic-v1` | `.../tensorflow_datasets/clic` |

Use the [dataset catalog](../datasets/catalog.md) to pair names and versions. Hit inputs require the dedicated `*-hits-v1` recipes and substantially more memory.

## Next steps

- [Continue or fine-tune](resume-and-fine-tune.md) from a checkpoint.
- Run the `test` command and detector-level plots described in [Key4HEP validation](../validation/key4hep.md).
- Compare PyTorch and ONNX execution with [ONNX validation](../validation/onnx.md).
