# Export and validate ONNX

ONNX validation checks that a deployment-oriented export agrees numerically with the PyTorch model and measures its runtime and memory behavior. The historical script is named `cms-validate-onnx.py`, but it reads TFDS and is also used for CLD and CLIC validation.

## Required files

Keep these inputs from the same experiment:

- `checkpoints/checkpoint-*.pth`, containing model weights and training state;
- `model_kwargs.pkl`, containing the resolved model and data configuration; and
- a compatible TFDS dataset and version.

The complete deployment bundle includes the checkpoint and `model_kwargs.pkl`, which supplies the architecture, feature, padding, and output choices needed to reconstruct the model.

## CPU numerical check

Use the CPU ONNX Runtime environment from the [installation guide](../getting-started/installation.md). This example compares ten CLD events:

```bash
uv run --project envs/ort-cpu python3 scripts/cms-validate-onnx.py \
  --checkpoint /path/to/experiment/checkpoints/checkpoint-2000.pth \
  --model-kwargs /path/to/experiment/model_kwargs.pkl \
  --dataset cld_edm_ttbar_pf/1 \
  --data-dir data/tfds/tensorflow_datasets/cld \
  --num-events 10 \
  --device cpu \
  --configs PT_ATTN_MATH_FP32 ONNX_ATTN_MATH_FP32 \
  --num-threads 1 \
  --outdir onnx_validation/cld-cpu
```

Success creates `summary.json`, exported `.onnx` files, and diagnostic plots below the output directory. Inspect the summary for exceptions, non-finite values, and PyTorch/ONNX differences before interpreting timing.

## GPU and precision comparisons

Use `envs/ort-gpu` and `--device cuda` only on a compatible Nvidia system. Available configuration names are printed by:

```bash
uv run --project envs/ort-gpu python3 scripts/cms-validate-onnx.py --help
```

FP16 can improve throughput and reduce memory, but it has different numerical tolerances from FP32. Report precision and attention implementation with every result. Fused attention is an export/runtime option and must be compared against an appropriate PyTorch reference.

HEPT, HEPTv2, GNNLSH, and LitePT have architecture-specific padding and operator constraints. The current `--help` lists the supported validation configurations. LitePT remains Nvidia-oriented. For hash/block models, pad to a compatible multiple and record it.

## Variable event sizes and padding

Particle-flow events contain different numbers of detector elements. The validator groups or pads events for execution. Padding changes memory and timing even when physics outputs are masked correctly. Record the event-size distribution, `--pad-bin-size` or model block size, batch policy, warm-up, and whether compilation or caching is included.

## Benchmark methodology

First establish numerical agreement on representative small, medium, large, and pathological events. Then benchmark after warm-up using enough repetitions to quote a timing distribution. Record hardware, software versions, threads, device, precision, model, event sizes, padding, and peak memory.

For a collection of validation directories, create comparison plots with:

```bash
uv run --project envs/ort-cpu python3 scripts/plot-onnx-summary.py \
  --indir onnx_validation \
  --outdir onnx_validation/plots
```

Normalize or stratify event-size and padding distributions before comparing throughput across runs.
