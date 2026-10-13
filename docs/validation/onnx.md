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

For attention models, the new export uses **standard ONNX `Attention` opset 24**
with `is_causal=0` and `nonpad_kv_seqlen: int64[B]`. The inputs remain
`Xfeat_normed[B,L,F]` and `mask[B,L]`, with dynamic batch and sequence dimensions.
Each mask must be binary, nonempty, and right-padded. Holes, left padding, and
empty events are rejected by the validation tools; external callers must enforce
the same contract. Lengths exclude padded keys, and the backbone masks padded
query outputs. Final prediction heads may produce nonzero values on padding;
consumers must discard those positions.

Use `mlpf.model.onnx_attention.export_mlpf` for programmatic export. It traces at
opset 20 (the legacy PyTorch exporter cannot handle the full model directly at
24), upgrades standard operators with ONNX's version converter, and lowers its
typed attention placeholders to `Attention-24`. The final file contains no custom
MLPF operators or global SDPA overrides. Other architectures and decomposed math
attention retain opset 20.

The relevant configurations are:

| Configuration | Execution |
| --- | --- |
| `PT_ATTN_MATH_FP32` | Dense masked FP32 reference |
| `PT_ATTN_FLASH_FP16` | Packed variable-length PyTorch FlashAttention under FP16 autocast |
| `ONNX_ATTN_OP_FP32` | Standard `Attention-24` in FP32; **not** FlashAttention |
| `ONNX_ATTN_FLASH_FP32_FP16` | FP32 model with FP16 Q/K/V at the attention boundary |
| `ONNX_ATTN_FLASH_FP16` | Full FP16 model with `Attention-24` |
| `ONNX_ATTN_MATH_FP32` / `FP16` | Decomposed attention with explicit key masking |

`ONNX_ATTN_OP_FP32` replaces the misleading `ONNX_ATTN_FLASH_FP32` name.
GPU Flash configurations require CUDA; the script no longer substitutes CPU math
attention for a requested Flash run. PyTorch Flash uses the packed backbone,
while ONNX keeps padded tensors and passes lengths. Their learned weights and
attention semantics are shared. Existing checkpoints trained with unmasked
padding need physics validation after this correction.

```bash
uv run --project envs/ort-gpu python3 scripts/cms-validate-onnx.py \
  --checkpoint /path/to/experiment/checkpoints/checkpoint-2000.pth \
  --model-kwargs /path/to/experiment/model_kwargs.pkl \
  --dataset cld_edm_ttbar_pf/1 --data-dir /path/to/tfds/cld \
  --num-events 12 --batch-size 4 --pad-bin-size 512 --device cuda \
  --configs PT_ATTN_MATH_FP32 PT_ATTN_FLASH_FP16 ONNX_ATTN_FLASH_FP16 \
  --save-inputs --profile --outdir onnx_validation/cld-cuda
```

Flash ONNX runs request `sdpa_kernel=1` and print kernel diagnostics. Verify
`SdpaKernel=FLASH_ATTENTION`; the provider option alone does not prohibit every
fallback. ORT profiles verify provider placement, but a CUDA node alone does not
prove FlashAttention. An explicit attention mask would disable the Flash path;
the exporter passes lengths instead. `--compile` optionally enables PyTorch
compilation. CUDA memory reported by the model validator is PyTorch allocator
memory and does **not** include ORT allocations; use the microbenchmark's memory
sampler for process/GPU memory comparisons.

HEPT, HEPTv2, GNNLSH, and LitePT have architecture-specific padding and operator constraints. The current `--help` lists the supported validation configurations. LitePT remains Nvidia-oriented. For hash/block models, pad to a compatible multiple and record it.

## Variable event sizes and padding

Particle-flow events contain different numbers of detector elements. The validator groups or pads events for execution. Padding changes memory and timing even when physics outputs are masked correctly. Record the event-size distribution, `--pad-bin-size` or model block size, batch policy, warm-up, and whether compilation or caching is included.

`--batch-size` batches distinct events; the last batch may be smaller. Every
export is traced with two unequal-length events, independently of the requested
inference batch size. Summary entries represent **one batch invocation**:
`runtime` is seconds per batch, `size` is total valid elements, and `lengths`,
`batch_size`, `event_indices`, and `padded_length` retain the individual event
sizes and padding cost. `--save-inputs` writes `inputs_*.npz` once per batch for
replay without TFDS or a PyTorch checkpoint.

## AMD and standalone runtime checks

`envs/ort-amd` installs AMD's vendor MIGraphX wheel and is inference-only. On a
Linux x86-64 host with ROCm/MIGraphX 7.2.4, run:

```bash
uv sync --project envs/ort-amd
uv run --project envs/ort-amd python3 scripts/validate-onnx-runtime.py \
  --model onnx_validation/cld-cpu/model_math_fp32.onnx \
  --inputs onnx_validation/cld-cpu/inputs_*.npz \
  --provider migraphx --outdir onnx_validation/cld-amd
```

Produce these inputs with `--save-inputs` in the CPU example above. Start with
the opset-20 math export: support and Flash lowering of `Attention-24` in
MIGraphX have **not** been established. This environment uses vendor ORT 1.23.2,
not the 1.29.0 CPU/CUDA runtime. No AMD hardware was available for validation.

The replay script compares valid outputs to the same model on ORT CPU, rejects
non-finite or out-of-tolerance results, records first-call versus repeated-call timing, and saves
a provider profile. `--require-full-gpu` also fails on CPU compute fallback.
Use `--provider-option key=value` for explicit EP settings. Importing the wheel
and listing MIGraphX does not prove the GPU provider can initialize. Fixed shape
buckets are still important for MIGraphX compilation cost; its warm timings
exclude the first call for each input batch (a previously seen shape may already
be cached).

The same replay script works with `envs/ort-cpu --provider cpu` or
`envs/ort-gpu --provider cuda`.

## Attention regression tests and microbenchmark

```bash
uv run --project envs/ort-cpu python3 -m pytest tests/test_attention24_export.py
uv run --project envs/ort-gpu python3 -m pytest tests/test_attention24_export.py
ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1 uv run --project envs/ort-gpu \
  python3 scripts/benchmark-sdpa-onnx.py --device cuda --batch-size 3 \
  --seq-lens 128 512 --modes masked --num-heads 16 --head-dim 16 \
  --variants pt_flash_fp16 ort_attention24_fp16 --provider-option sdpa_kernel=1 \
  --profile --outdir onnx_validation/attention-microbenchmark
```

The regression tests check fresh full-model exports with different batch sizes
and lengths, separate-event versus batched inference, padding contamination,
FP32/mixed/FP16 precision, and CUDA Flash dispatch in both runtimes. GPU tests
skip when the required hardware or provider is absent. The microbenchmark keeps
the old `ort_mha_*` variants as comparisons; `ort_attention24_*` uses the new
length-aware operator and varying per-event lengths.

Upgrade validation (2026-10-13): the CPU model/masking/export suite passed
26 tests with 3 skips, and the CUDA regression confirmed FlashAttention in both
PyTorch and ORT. A small randomly initialized CLD model also completed the TFDS
validator on three real events, batched as two plus one with different padding
lengths. CPU FP32 mean absolute error was about 1e-7; CUDA mixed precision was
about 4e-5 against PyTorch FP32. Mixed-precision saved-input replay passed the
default tolerances. Full FP16 replay exceeded them on some outputs (maximum
absolute error about 0.006); tolerances and physics performance must be assessed
with a trained checkpoint. The AMD environment installed and replayed the math
export on CPU; AMD GPU execution remains untested.

## Benchmark methodology

For batch-size scaling, the attention microbenchmark accepts `--batch-sizes`
and `--length-policy balanced`. With `--mask-fraction 0.5`, balanced lengths
sample the interval from 50% to 100% of the padded length at bin midpoints;
the mean is 75% (up to integer rounding), independently of batch size. This
avoids changing the average event size while measuring batching. For example,
at padded length 512, batch size four has valid lengths 480, 416, 352, and 288.
The single-event baseline has length 384.

```bash
uv run --project envs/ort-cpu python3 scripts/benchmark-sdpa-onnx.py \
  --device cpu --batch-sizes 1 2 4 8 16 --seq-lens 512 1280 \
  --length-policy balanced --modes masked --mask-fraction 0.5 \
  --num-heads 16 --head-dim 16 --num-threads 1 --warmup 3 --iters 10 \
  --variants pt_math_fp32 ort_attention24_fp32 ort_mha_fp32 \
  --outdir onnx_validation/batch-cpu

uv run --project envs/ort-gpu python3 scripts/benchmark-sdpa-onnx.py \
  --device cuda --batch-sizes 1 2 4 8 16 --seq-lens 512 1280 \
  --length-policy balanced --modes masked --mask-fraction 0.5 \
  --num-heads 16 --head-dim 16 --warmup 5 --iters 30 \
  --variants pt_flash_fp16 ort_attention24_fp16 ort_mha_fp16 \
  --provider-option sdpa_kernel=1 --provider-option use_tf32=0 \
  --outdir onnx_validation/batch-gpu

uv run --project envs/ort-cpu python3 scripts/plot-sdpa-onnx-summary.py \
  --cpu-csv onnx_validation/batch-cpu/summary.csv \
  --gpu-csv onnx_validation/batch-gpu/summary.csv \
  --sweep batch --outdir onnx_validation/batch-plots
```

The plotter produces separate median latency, event throughput, valid-element
throughput, and process-memory figures (PNG/PDF). Panels separate device,
precision, and workload; implementation colors, markers, and dashes remain
consistent. Use the default `--sweep sequence` for sequence-length scans.
The adjacent summary JSON supplies metadata omitted by older CSV files.

Timing excludes memory-sampler cleanup. GPU ORT calls include host/device
transfers through `session.run`; PyTorch inputs already reside on the GPU, and
its variable-length path includes packing/unpacking. These measurements are
call-level timings, not isolated kernel benchmarks. Process-memory samples
include runtime overhead and retained allocations, not just attention workspace.

First establish numerical agreement on representative small, medium, large, and pathological events. Then benchmark after warm-up using enough repetitions to quote a timing distribution. Record hardware, software versions, threads, device, precision, model, event sizes, padding, and peak memory.

For a collection of validation directories, create comparison plots with:

```bash
uv run --project envs/ort-cpu python3 scripts/plot-onnx-summary.py \
  --indir onnx_validation \
  --outdir onnx_validation/plots
```

Normalize or stratify event-size and padding distributions before comparing throughput across runs.
