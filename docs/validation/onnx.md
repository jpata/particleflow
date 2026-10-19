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

ORT 1.29 places `Elu-22` on CPU in the opset-24 full-model export, including
when CUDA runs `Attention-24`. The fused export lowers ELU into standard
`Less`, `Exp`, `Sub`, and `Where` operations so that activations stay on CUDA.
Provider profiling remains necessary: a CUDA session can still assign individual
nodes to CPU.

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

`envs/ort-amd` installs ORT 1.29.0 plus AMD's MIGraphX provider plugin 1.0.0
and MIGraphX 2.18.0 (ROCm 10.1.0 builds). It is inference-only. On a Linux
x86-64 host with ROCm 10.1.0 and a supported AMD GPU, run:

```bash
uv sync --project envs/ort-amd
uv run --project envs/ort-amd python3 scripts/validate-onnx-runtime.py \
  --model onnx_validation/cld-cpu/model_math_fp32.onnx \
  --inputs onnx_validation/cld-cpu/inputs_*.npz \
  --provider migraphx --outdir onnx_validation/cld-amd
```

Produce these inputs with `--save-inputs` in the CPU example above. Start with
the opset-20 math export: support and Flash lowering of `Attention-24` in
MIGraphX have **not** been established. All three environments now pin ORT
1.29.0. The script explicitly registers the AMD plugin and selects its first
available device using ORT's plugin API. It records plugin versions in the
summary and refuses initialization failure or whole-session CPU fallback.
No AMD hardware was available for validation; local checks cover package
installation and CPU execution of both math and Attention-24 exports. The host
also lacks ROCm, so plugin registration cannot complete locally.

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
FP32/mixed/FP16 precision, CUDA Flash dispatch in both runtimes, and the absence
of `Elu` nodes in the fused export. GPU tests skip when the required hardware or
provider is absent. The microbenchmark keeps
the old `ort_mha_*` variants as comparisons; `ort_attention24_*` uses the new
length-aware operator and varying per-event lengths.

Use `--export-shapes both` with the benchmark to compare symbolic batch/sequence
dimensions against a separate fixed-shape ONNX export for every `(batch, length)`
workload. `--export-shapes static` runs only fixed-shape exports; the default
remains `dynamic`. Results record `export_shape` in JSON/CSV, and static variants
have `_static` in their names. `--keep-models` retains graphs for shape inspection.

```bash
uv run --project envs/ort-gpu python3 scripts/benchmark-sdpa-onnx.py \
  --device cuda --batch-size 4 --seq-lens 512 --modes masked \
  --variants ort_attention24_fp16 ort_mha_fp16 \
  --provider-option sdpa_kernel=1 --export-shapes both \
  --outdir onnx_validation/attention-static-comparison
```

The hand-built `ort_attention24_*` microbenchmark does not contain the full
model's Q/K/V reshape chains, so this comparison does not measure their CPU
shape-node overhead.

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

To compare multiple GPU runs, pass each result as `--input LABEL=CSV` instead
of the CPU/GPU flags, then run the plotter once for each sweep.

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

### Upgrading the deployed July 2025 CMS artifact

`scripts/convert-cms-onnx-attention24.py` converts the pinned
`mlpf_5M_attn2x3x256_bm12_relu_checkpoint10_8xmi250_fp32_fused_20250722.onnx`
artifact. It verifies the source SHA256, preserves every initializer byte, replaces
six custom attention calls with standard Attention-24, and replaces the broken
padding ScatterND with Where. FP16 attention boundaries are retained for CUDA
FlashAttention; the other model operations and public tensors remain FP32.

```bash
uv run --project envs/ort-cpu python scripts/convert-cms-onnx-attention24.py \
  original.onnx attention24.onnx
ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1 \
uv run --project envs/ort-gpu python scripts/validate-cms-attention24.py \
  --original original.onnx --model attention24.onnx \
  --inputs cms_events --device cuda --output validation_cuda.json
uv run --project envs/ort-cpu python scripts/validate-cms-attention24.py \
  --original original.onnx --model attention24.onnx \
  --inputs cms_events --device cpu --limit 4 --output validation_cpu.json
```

Each input NPZ contains one unpadded FP32 `Xfeat_normed` array `[1,L,55]`,
prepared with the CMS feature ordering in `mlpf/heptfds/cms_pf/cms_utils.py`.
The validator compares valid outputs against original single-event inference,
using batch sizes 1, 2, and 4 with unequal lengths and several padding sizes.
It records numerical differences and classification changes; completion alone
is not a numerical acceptance test. Check the JSON metrics and CUDA kernel log.
Masks must be binary with a valid prefix followed by right padding. Discard
padded outputs. This conversion does not support masks with internal holes.

The original CPU runtime effectively uses FP32 attention whereas CUDA uses
FP16 FlashAttention. The converted artifact prioritizes the latter; CPU results
can differ appreciably from the original CPU model. Assess each backend against
its own original reference. CUDA validation does not establish MIGraphX support
or replace a full downstream physics validation.

### W7900 / MIGraphX validation (2026-10-16)

The Radeon PRO W7900 (`gfx1100`) was tested with PyTorch
`2.9.1+rocm7.2.1.gitff65f5bc`, ROCm 7.2.4, MIGraphX 2.15.0 and the
monolithic ONNX Runtime MIGraphX 1.23.2 wheel. The AMD driver/module version
was 6.16.13. This is a different stack from `envs/ort-amd`'s ORT 1.29 plugin
environment; these results do not validate that newer environment.

Select MIGraphX explicitly; PyTorch names AMD devices `cuda` too:

```bash
python scripts/benchmark-sdpa-onnx.py \
  --device cuda --ort-provider migraphx --require-full-gpu \
  --variants pt_math_fp32 pt_math_fp16 onnx_sdpa_export_fp32 onnx_sdpa_export_fp16 \
  --export-shapes static --seq-lens 128 512 --modes both \
  --warmup 2 --iters 5 --profile --outdir onnx_validation/w7900-math
```

Inspect each result's status: per-variant failures are recorded rather than
causing a nonzero process exit. `--require-full-gpu` disables CPU node fallback;
provider availability alone does not establish that every node runs on GPU.
AMD VRAM sampling is unavailable in this benchmark's NVIDIA memory sampler.

On this stack, unmasked legacy MATH exports execute on MIGraphX. Masked
exports contain an `IsNaN` safe-softmax guard that requires CPU fallback,
including transfers of the full attention-probability tensor. They fail with
strict GPU execution enabled. Removing that guard requires preserving the
behavior of queries with no valid keys; multiplying NaN outputs by zero is
insufficient.

For the tested FP16 decompositions, setting
`MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention` before starting Python enables
rocMLIR attention fusion. The unmasked compiled program changes from three
GPU kernels to one and avoids the full attention-score intermediate. FP32
compilation with this override fails on the tested shapes, while default
FP32 decomposition runs. This setting does not remove the masked `IsNaN`
fallback. It is distinct from CUDA's `sdpa_kernel` provider option.

Attention-24 is rejected by this ORT 1.23.2 stack before execution; CUDA
Attention-24 validation must not be interpreted as W7900 support. These are
attention microbenchmark findings, not full-model throughput or physics
validation.

### Explicit attention export profiles

Use `mlpf.model.onnx_attention.export_mlpf` for both representations; choosing
an eager PyTorch attention backend does not select an ONNX runtime kernel.
For an FP32 eval model and finite FP32 example features/mask:

```python
import torch
from mlpf.model.onnx_attention import export_mlpf

export_mlpf(model, features, mask, "attention24.onnx",
            attention_format="attention24", attention_dtype=torch.float16)
export_mlpf(model, features, mask, "decomposed.onnx",
            attention_format="decomposed", attention_dtype=torch.float16)
```

The first representation uses standard Attention-24 and valid key lengths,
without a dense attention mask. The second uses explicit scaled
MatMul/FP32 Softmax/MatMul operations, with a broadcast key mask. Power-of-two
Q prescaling reduces FP16 dot-product overflow risk; it does not guarantee
safety for unbounded inputs. In both cases the other model operations and
public I/O retain their model/example dtypes. This exports the same checkpoint
without changing learned tensors. Omitted `attention_format` preserves the
legacy `export_onnx_fused` selection. Export temporarily overrides native
Flash/jagged execution and restores the model's settings, including on failure.

Both profiles require binary `[B,L]` masks with nonempty valid prefixes and
right padding. Callers must validate masks at inference too: the ONNX graph
cannot throw for invalid masks. Padded input features are sanitized and padded
predictions are zero. Nonfinite cleanup is lowered to equivalent comparisons,
and four final masked heads preserve the named outputs as separate materialized
tensors, avoiding strided Split views at the GPU runtime boundary. This helper
exports elementwise predictions; set-decoder outputs need a different mask
contract and are rejected explicitly.

The CMS validator exposes the mixed profiles as `ONNX_ATTN_FLASH_FP32_FP16`
and `ONNX_ATTN_MATH_FP32_FP16`. The latter can be exported/tested on CPU before
replay on AMD. `cms-validate-onnx.py --device cuda` itself selects CUDA EP;
use `scripts/validate-onnx-runtime.py --provider migraphx` for AMD replay.

### CPU CI, offline compilation, and device qualification

Ordinary GitHub CI runs both formats through ONNX checker and ORT 1.30 CPU
inference. The lightweight environment does not need TensorFlow, flash-attn,
or a GPU. The workflow is `.github/workflows/attention-export.yml`.

```bash
mkdir -p onnx_validation
PYTHONPATH=. python -m pytest \
  tests/test_attention24_export.py tests/test_attention_export_profiles.py \
  -q --basetemp=onnx_validation/export-ci
```

Tests cover distinct events and changed batch/sequence dimensions, poisoned
padding, zero padded outputs, unchanged learned tensors, restored export state,
invalid masks, large FP16 scores, and IEEE NaN/infinity cleanup. Native Flash
configuration can also be exported on CPU. These checks establish graph
semantics, not execution by CUDA/MIGraphX or GPU fusion.

Device tests are opt-in and fail if the explicitly requested provider cannot
initialize. They use CPU PyTorch for export/reference, so GPU-enabled PyTorch
and flash-attn are not prerequisites. Install the appropriate ORT provider and
native GPU libraries. The qualification runtime must be ORT 1.30 or newer:

```bash
mkdir -p onnx_validation
ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1 MLPF_TEST_PROVIDER=cuda \
PYTHONPATH=. python -m pytest tests/test_attention_export_profiles.py \
  -k real_device -q --basetemp=onnx_validation/export-cuda

MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention MLPF_TEST_PROVIDER=migraphx \
PYTHONPATH=. python -m pytest tests/test_attention_export_profiles.py \
  -k real_device -q --basetemp=onnx_validation/export-migraphx
```

For a separately built MIGraphX plugin (including a CMSSW build), set
`MLPF_TEST_MIGRAPHX_LIBRARY=/path/to/the/provider-plugin.so` in the same command.
This loads that exact library without requiring AMD's Python plugin wheel.
Monolithic MIGraphX providers are also supported.

The tests disable CPU compute fallback, check provider placement and numerical
batching/padding tolerances, and retain models, profiles and `validation.json`
under `--basetemp`. Reports include package versions, available driver-tool
output and loaded-library paths. CUDA must also report FlashAttention dispatch.
MIGraphX placement alone is not fusion evidence. Add
`MLPF_TEST_REQUIRE_FUSION=1` to the AMD command to collect compiler traces and
require attention code objects in the last compiled program, excluding earlier
tuning/pass dumps. This retains a potentially large `compile.log`. Without
that flag, the test makes no MIGraphX fusion claim. A rocprof trace can provide
additional evidence of the actual kernel dispatches.

MIGraphX 2.18+ additionally documents [hardware-free GPU cross-compilation](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/develop/dev/cross_compilation.html).
With the driver and its userspace dependencies installed, opt into the offline
compiler/fusion check:

```bash
MLPF_TEST_MIGRAPHX_DRIVER=/opt/rocm/bin/migraphx-driver \
PYTHONPATH=. python -m pytest tests/test_attention_export_profiles.py \
  -k offline_compile -q --basetemp=onnx_validation/export-offline
```

Local qualification on 2026-10-19 used the AMD wheel
`migraphx-libs==2.18.0+rocm10.1.0` (driver reports
`2.18.0.20250912-17-754-g95672916e`) and the ROCm 10.1.0
`rocm-sdk-core`, `rocm-sdk-libraries`, and `rocm-sdk-devel` userspace packages
unpacked under `/tmp`. The driver needs the core and library directories on
`LD_LIBRARY_PATH` even for offline compilation: its executable links HIP,
MIOpen, rocBLAS, and hipBLASLt. This does not mean those library backends can
execute without a GPU. The test's named dimension arguments must use
`--input-dim @Xfeat_normed 2 128 55 --input-dim @mask 2 128`.

With those arguments, the driver parses the exported model and forms MLIR
attention regions, but this packaged build aborts during `gpu::lowering` with
`get_device_id: No device`. The offline test therefore **has not passed** on
this CPU host. Intermediate MLIR regions are not final compiled-kernel or
execution evidence. A compiler build that completes this model's device-free
lowering is still required; the test retains this failure rather than skipping
it. No host GPU driver or system ROCm installation was changed.

This compiles a static test shape for `gfx1100`, checking that a fused attention
code object is present. It does not allocate GPU buffers, run kernels, test ORT
EP integration, or validate GPU numerical results. Cross-compilation supports
rocMLIR/MIGraphX-generated kernels; library backends such as rocBLAS and MIOpen
are unsupported. The test clears their conflicting selection flags and saves
`compile.log`. It requires an appropriate recent driver and is not installed
or run by the ordinary CPU workflow.

There is no supported CUDA simulator used by this CI. CPU inference validates
the exported Attention-24 semantics; actual CUDA FlashAttention dispatch still
requires a real GPU. Do not replace device qualification with a CPU fallback
and report it as a GPU pass. Trained checkpoints also need their own numerical
and physics validation: passing the small deterministic test model is not a
claim that shape-dependent FP16 differences disappear in deployed models.

### HEPTv2 deterministic hashing

HEPTv2 uses discrete bucket assignments. Small differences in floating-point
`atan2` or projection results can change sort ranks and therefore the model's
predictions. The `model.heptv2.deterministic_hashing` option (default `true`)
uses a shared integer path in PyTorch and ONNX:

- Eta is clipped to [-32, 32] and rounded on a 2^-20 grid.
- Phi uses 26 integer CORDIC iterations and the same 2^-20 grid; padded zero
  vectors map to zero. Input sine/cosine components are clipped to [-32, 32].
- Projection coefficients are clipped to [-16, 16] and rounded on a 2^-12
  grid. Projections and region arithmetic use int64.
- Stable ascending sorts resolve equal keys by input order. Padding uses a
  finite integer sentinel, with a one-unit gap between hash regions.

This changes bucket assignments and quantizes the coordinates used by the
positional encoding. Existing checkpoint predictions can change; evaluate or
retrain checkpoints before deployment. Setting the option to `false` selects
floating-point coordinates and projections, but does not promise exact
historical tie ordering or cross-runtime agreement.

`tests/test_heptv2_export.py` checks bitwise coordinate agreement, duplicate
coordinates, numerical outputs at unseen batch/sequence sizes, and padding
isolation. `scripts/qualify-heptv2-onnx.py` exports a seeded model and replays
saved input/reference bundles on CPU, CUDA, or MIGraphX. Its real-event mode
uses CMS TFDS inputs, but its weights are untrained: these checks establish
export correctness, not physics performance. GPU qualification additionally
requires profiling evidence of GPU placement; a CPU fallback is diagnostic.

The earlier deterministic bundle passed CPU comparisons for three synthetic
and three real CMS events (maximum absolute output difference below 1e-5).
Initial NGT GPU qualification on 2026-10-19 **failed** for this hashing path.
The original bundle has model SHA256
`a833449f0ed1d4d0522eaea0b9d8b3458b394538b082584512db79f74a09ca57`.
Results and profiles are retained under
`onnx_validation/heptv2-qualification/ngt-gpu-current/`.

- NVIDIA L40S, driver 590.48.01, ORT GPU 1.29.0: dynamic export fails
  initialization when CPU fallback is disabled. A diagnostic fallback run
  records 24,624 CUDA and 9,384 CPU node executions. Static specialization
  runs all six cases with CUDA-only profiled placement, but numerical checks
  fail for real events 1 and 2. Maximum absolute momentum errors are 0.76825
  and 1.27056 respectively; the same failures occur in the dynamic diagnostic
  run. Three synthetic cases and real event 0 pass (maximum error 6.20e-6).
- AMD W7900, native ORT MIGraphX 1.23.2 with ROCm/MIGraphX 7.2.4:
  both dynamic and static graphs compile with CPU fallback disabled, then
  fail on the first inference in `migraphx_execution_provider_utils.h:351`,
  `ToInteger: invalid input for conversion to integer`. There is no numerical
  or successful GPU-execution qualification for these runs.
- The separately installed ORT 1.29.0 / MIGraphX plugin
  `1.0.0+rocm10.1.0` alternative fails to load with undefined symbol
  `migraphx_compile_options_set_advance_backend_options` against the tested
  library search path. This is an environment failure, not model evidence.

The CUDA failure was traced to TopK tie ordering: CPU and CUDA hash coordinates
matched exactly, but 1,304 eta ranks and 1,334 phi ranks differed for the
3,900-element real event. Export now first identifies equal-value groups and
then sorts unique `(group, original input index)` integer keys. These keys
are bounded by sequence length squared, avoiding overflow from multiplying
large hash values or the padding sentinel by sequence length. Both required
TopK outputs remain connected through an exact integer zero dependency;
otherwise older MIGraphX EPs expose unused values as extra partition outputs
and generate output names their runtime cannot parse.

The corrected bundle has SHA256
`46995e541a88d61a27c416ddcb08f17026b15b634e6f6a31f3f26218c627b0c7`.
Both devices now pass all six static FP32 cases: synthetic shapes
`[2,200,55]`, `[1,300,55]`, `[3,400,55]`, and three real CMS events with
sequence lengths 4100, 3900, and 3400. CPU fallback is disabled, every
output comparison passes, padded predictions are zero, and all six profiles
on each device contain only the requested GPU provider.

| Device / runtime | Cases passed | Maximum absolute output error |
| --- | --- | --- |
| NVIDIA L40S / ORT CUDA 1.29.0 | 6/6 | 7.16e-6 |
| AMD W7900 / native ORT MIGraphX 1.23.2 + MIGraphX 2.18 | 6/6 | 7.96e-6 |

Artifacts are under `onnx_validation/heptv2-qualification/ngt-fixed/`:
`final-bundle/`, `l40s-final/summary.json`, `w7900-final/summary.json`,
profiles, loaded-library evidence, reduced reproducers, and source hashes.
A fresh export from the repaired source reproduced the model SHA256 exactly.

The native MIGraphX 2.15 compiler in ROCm 7.2.4 additionally lowers int64
hashing to FP32 and produces incorrect predictions. Disabling its
`eliminate_data_type` pass exposes unsupported int64 GPU kernels. This older
stack cannot qualify the deterministic export. The repaired AMD setup uses
native ORT MIGraphX 1.23.2 with the newer MIGraphX 2.18 wheel libraries and
ROCm SDK userspace libraries. Those libraries must precede system ROCm in
`LD_LIBRARY_PATH`; the loaded library paths are retained with the results.
The separate ORT 1.29 plugin swapped eta/phi outputs in a reduced two-output
reproducer, so it is not the qualified provider. The provider helper also
reuses registered plugin devices across sessions and uses basic ONNX
optimizations to avoid shape-dependent extended CPU fusions.

The graph now masks each prediction head separately. Returning Split views
of a shared concatenated allocation exposed incorrect strided output binding
and buffer corruption on AMD; materialized outputs keep the four public names
and widths. A mixed-width regression checks routing and padding at batch 3,
sequence 400, and can run on the requested GPU provider.

For qualification, `ngt/run-heptv2-onnx.sh` disables `eliminate_data_type`
before starting Python and retains `MIGRAPHX_GPU_OPTIMIZE=3`, adding
`MIGRAPHX_GPU_HIP_FLAGS="-mllvm -enable-gvn-memdep=false"`. LLVM's GVN
memory-dependence analysis otherwise spends minutes compiling full integer
sorts above 4096 elements. The targeted compiler workaround passed a minimal
4,100-element sort bitwise, including keys of magnitude 2^44 and the 2^60
padding sentinel. Model inputs and numerical tolerances are unchanged;
this run makes no optimized-throughput claim.

The sort and output-routing regressions are also runnable on an actual GPU with
`MLPF_TEST_PROVIDER=cuda` (or `migraphx` in a configured integer-capable stack):

```bash
PYTHONPATH=. MLPF_TEST_PROVIDER=cuda python -m pytest \
  tests/test_heptv2_export.py -k 'exported_sort or materializes' -q
```

Five CUDA sort/output regressions and 60 CPU export/runtime checks pass
(three device-only checks skipped in the CPU suite). The CPU workflow includes
the HEPTv2 and runtime tests. Dynamic exports can
still require CPU shape operations; use static specialization for strict GPU
qualification. These seeded-weight checks do not validate trained physics
performance or establish attention-kernel fusion.
