# Full-model export evidence protocol

Use this protocol to qualify an elementwise attention or HEPTv2 export on a specific runtime. Start with the [ONNX guide](onnx.md) for the input contract and a CPU check. Qualification requires numerical agreement, provider-placement evidence, and reproducible measurements. A kernel dispatch or successful export alone is insufficient.

## Current qualification boundary

| Profile | Retained evidence | Interpretation |
|---|---|---|
| Portable FP32 attention and HEPTv2 | 42/42 trained-checkpoint cases pass across CPU, NVIDIA L40S, and AMD W7900. | Qualified for the recorded checkpoints, inputs, and runtime stacks. |
| CUDA Attention-24 with FP16 attention | GPU placement and fused kernels verified; 0/7 trained cases pass the FP32 tolerance. | Executable, but not numerically qualified under that criterion. |
| AMD decomposed FP16 attention with MLIR fusion | GPU placement and fused kernels verified; 0/7 trained cases pass the FP32 tolerance. | Executable, but not numerically qualified under that criterion. |
| Particle-set output | No export contract in `export_mlpf`. | Requires separate implementation and validation. |

The acceptance tolerance in these studies is `atol=1e-3, rtol=3e-3`. Evidence is artifact-specific; it does not establish reconstruction physics quality or qualify all model shapes and environments. Detailed measurements and provenance are retained below.

## Runtime and compiler requirements

The GPU evidence uses NVIDIA L40S with ORT CUDA 1.29 and AMD W7900 with native ORT MIGraphX 1.23.2, MIGraphX 2.18, and ROCm SDK libraries. This AMD stack differs from the packaged `envs/ort-amd` plugin environment. Follow [the GPU tooling guide](../../ngt/README.md) for exact library paths and worker setup; preserve outputs from temporary workers before they expire.

HEPTv2 needs integer-preserving hashing and sorting. The qualification launcher `ngt/run-heptv2-onnx.sh` disables `eliminate_data_type` and retains `MIGRAPHX_GPU_OPTIMIZE=3` with `MIGRAPHX_GPU_HIP_FLAGS="-mllvm -enable-gvn-memdep=false"`. These settings avoid unsupported integer lowering and excessive compiler time; they do not establish optimized throughput. Older MIGraphX 2.15 lowering to FP32 is unsuitable. Load the qualified libraries ahead of system ROCm and retain the actual loaded-library paths.

Use static specialization when full GPU placement is required. Dynamic exports can involve CPU shape operations. The provider helper rejects whole-session fallback; strict qualification additionally rejects profiled CPU compute nodes.

## Opt-in regression checks

The CPU workflow and its versions are described in [ONNX validation](onnx.md). Device profile tests require ORT 1.30 or newer and a configured provider. They use CPU PyTorch for export/reference and fail if a requested GPU provider cannot initialize:

```bash
mkdir -p onnx_validation
ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1 MLPF_TEST_PROVIDER=cuda \
PYTHONPATH=. python -m pytest tests/test_attention_export_profiles.py \
  -k real_device -q --basetemp=onnx_validation/export-cuda

MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention MLPF_TEST_PROVIDER=migraphx \
PYTHONPATH=. python -m pytest tests/test_attention_export_profiles.py \
  -k real_device -q --basetemp=onnx_validation/export-migraphx
```

`MLPF_TEST_MIGRAPHX_LIBRARY` can select a separately built provider library. Tests retain graphs, profiles, versions, loaded-library evidence, and `validation.json`. CUDA qualification requires Flash dispatch; AMD fusion additionally requires `MLPF_TEST_REQUIRE_FUSION=1` and attention code objects in the final compiled program. Provider placement alone does not establish fusion.

An offline MIGraphX compiler check is available with `MLPF_TEST_MIGRAPHX_DRIVER=/path/to/migraphx-driver` and `pytest tests/test_attention_export_profiles.py -k offline_compile`. It needs a compatible cross-compilation toolchain; its availability is not a claim of successful hardware execution. HEPTv2 GPU sort/output regressions use `MLPF_TEST_PROVIDER=cuda` or `migraphx` with `pytest tests/test_heptv2_export.py -k 'exported_sort or materializes'`.

## Baseline matrix

Run each model on CUDA/L40S, MIGraphX/W7900, and CPUExecutionProvider on the
L40S host (AMD EPYC 9534). Use FP32 decomposed attention, static specialization,
and identical ONNX model weights, inputs, masks and PyTorch references on all
three providers. This isolates a portable baseline. A separate seven-case
Attention-24/FP16 sweep on L40S uses the same model weights, input arrays and
FP32 references, with CUDA sdpa_kernel=1 requested. Public inputs/outputs remain
FP32. Keep these measurements labeled separately from the 42-case FP32 matrix.
An independent ORT-only Nsight trace at B=1,N=4000 confirms actual
onnxruntime::flash::flash_fwd_kernel execution. A second seven-case optimized
profile uses decomposed FP16 attention on W7900 with
MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention. A matching ROCm SDK rocprofv3 trace
verifies fused score/mask/softmax/value kernels at N=4000, and every shape's
final compiled program must contain five fused attention code objects with
no quadratic attention-score allocations. Both optimized profiles retain
FP32 public inputs/outputs and the exact same FP32 PyTorch reference arrays.

The controlled-sweep models use seed 42 and untrained weights. Both have three backbone
layers and eight heads; attention head dimension is 16, while HEPTv2 uses
embedding dimension 128, width 512, block size 100 and three hashes. Full
configuration is retained in each bundle manifest. These are export correctness
and implementation scaling measurements, not a trained-model physics comparison
or a claim that the two architectures have equal capacity. Repeat with selected
trained checkpoints before physics sign-off. In particular, deterministic HEPT
hashing can change predictions from older trained models.

## Inputs and numerical evidence

Use a fixed CMS feature pool from the existing real-event reference bundle,
repeat deterministically to the requested shapes, and right-pad 10% of each
event with zeros. Record padded shape and valid lengths separately. The sequence
sweep is B=1, N=100,500,1000,2000,4000; the batch sweep adds B=2,4 at N=1000.
Repeated features deliberately exercise tied HEPT hash coordinates. These
controlled inputs measure shape scaling; retain the previous three unmodified
CMS events as independent correctness evidence. They do not sample a physics
input distribution. Keep both real-event and controlled-sweep results.

For every output head, record max absolute error, MAE, RMSE, relative L2 error,
finite-output checks and exact zero padding. Compare valid outputs to eager
CPU PyTorch FP32 with atol=1e-3 and rtol=3e-3; publish actual errors rather than
only pass/fail. References are generated independently for each architecture.
Model/reference files are SHA256 checked before execution.

## Timing and memory

Use a fresh process per shape, model, provider and measurement mode. Run 10
warmups and 100 timed synchronous session.run calls, retaining every sample
and p50/p90/p95/p99. Inputs and outputs live on the host: these measurements
include transfers, allocation and ORT overhead. They exclude preprocessing,
reference calculation, model export and static specialization. Record session
creation and first inference separately, because AMD compilation can take
minutes and some compilation can occur lazily. Do not interpret a single
100-sample p99 as a precise tail-latency estimate.

Timing processes disable ORT profiling. A separate fresh process, with one
first call and one additional call, retains the placement trace. GPU fallback
is disabled in both modes. Require all profiled compute nodes on the requested
GPU provider. CPU fallback means a deliberately selected CPU-only session,
not hidden CPU partitions in a GPU measurement. Keep case exit codes and logs;
a missing result is a failed/incomplete measurement, never a zero latency.

Sample process RSS and whole-device used VRAM every 10 ms. NVIDIA uses NVML;
AMD reads mem_info_vram_used for the sole /dev/dri/card device allocated to the
pod. Record pre-session baseline, compilation/first-call peak and warm inference
peak separately, plus process lifetime RSS high-water mark. This captures ORT
allocations, including MIGraphX, which PyTorch allocator statistics omit. VRAM
is device-wide, so preserve device-idle evidence and avoid concurrent work.
Sampling can miss short transient peaks: sampled VRAM maxima are lower bounds,
and allocator reservation is included. Host RSS also includes the Python runtime,
model and loaded reference arrays; show absolute values and baseline changes.
Sampler overhead is included equally in timings and its 10 ms interval must
remain in the metadata. CPU timing uses one ORT intra/inter-op thread and records
CPU affinity. It is a reproducible fallback baseline, not a tuned CPU optimum.

Run CUDA first and then CPU on the same host; AMD runs on its separate host.
Record versions, CPU information, driver/device identity, compiler flags, source
hashes, model hashes and input hashes. AMD uses the qualified native
ORT 1.23.2 with MIGraphX 2.18/ROCm SDK libraries and the documented HEPT integer
and compiler workarounds. This is a comparison of complete software stacks.

## Reproduction and artifacts

`scripts/qualify-heptv2-onnx.py export --model-type attention|heptv2` generates
bundles; `--inputs-bundle` reuses CMS inputs, and `--sweep` plus `--batches`
generates the controlled shapes. `scripts/benchmark-model-export.py` measures
one case. `ngt/run-model-evidence.sh PROVIDER BUNDLE_ROOT RESULT_ROOT` runs the
matrix inside a configured runtime; see [the GPU tooling guide](../../ngt/README.md) for NVIDIA image and AMD
library setup. It bounds each worker to 30 minutes and retains each exit code.

Store each campaign under a distinct `onnx_validation/full-model-evidence/<campaign>/` directory:
model/reference bundles, raw results and memory samples, placement profiles,
logs, machine/source provenance, aggregate tables and scaling plots. The path
is ignored by Git; retain source scripts and this protocol in Git separately.
Run `scripts/summarize-model-evidence.py PACKAGE_ROOT --plots --require-complete`
and `scripts/render-model-evidence.py PACKAGE_ROOT` to build the final table,
CSV, PNG/PDF/SVG figures and portable HTML report. The acceptance gate includes
the supplementary profiles when their manifests are present (56 total cases).

## Controlled-sweep evidence

The seeded 2026-10-19 package is under
`onnx_validation/full-model-evidence/2026-10-19/`; open `report.html` for the
complete table and embedded plots. At B=1,N=4000 (3600 valid features), portable
FP32 median session.run latency was:

| Provider / device | Attention (ms) | HEPTv2 (ms) |
|---|---:|---:|
| CUDA / L40S | 16.18 | 12.89 |
| MIGraphX / W7900 | 13.36 | 13.69 |
| CPU / EPYC 9534, one ORT thread | 785.38 | 912.68 |

The separately labeled NVIDIA Attention-24/FP16 profile measured 2.18 ms at
that shape, with sampled VRAM above baseline of 584 MiB versus 2568 MiB for
decomposed FP32. Its maximum absolute error was 1.07e-3, passing the combined
absolute/relative tolerance; the actual FlashAttention kernel is retained in
`kernel-evidence/attention24-kernel-summary-bound.csv` and the Nsight report.
These are single-run complete-stack measurements with the qualifications above.

All 42 portable FP32 cases passed; the largest absolute output error across
that matrix was 9.12e-6. All seven supplementary FP16 cases also passed the
combined tolerance; their largest absolute error was 1.56e-3 (the B=4 case).
With the W7900 optimized profile, there are 56 independently retained placement
profiles and 5600 unprofiled latency samples. Exact executed static graphs are stored by SHA256, with
per-machine mappings verifying timing and placement used identical graphs.

## Optimized profile reproduction

Export NVIDIA with `--model-type attention --attention-format attention24
--attention-dtype float16 --inputs-bundle BUNDLE_ROOT/attention
--bundle BUNDLE_ROOT/attention24-fp16`. Inside its configured runtime, run:

```bash
CUDA_FLASH=1 bash ngt/run-model-evidence.sh cuda BUNDLE_ROOT RESULT_ROOT/supplementary attention24-fp16
```

Export AMD with `--model-type attention --attention-format decomposed
--attention-dtype float16 --inputs-bundle BUNDLE_ROOT/attention
--bundle BUNDLE_ROOT/attention-decomposed-fp16`. Inside the native ORT/MIGraphX
runtime with the documented LD_LIBRARY_PATH, run:

```bash
unset MIGRAPHX_DISABLE_MLIR MIGRAPHX_TRACE_COMPILE
MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention MIGRAPHX_DISABLE_PASSES= MIGRAPHX_GPU_HIP_FLAGS= \
TRACE_COMPILE_PLACEMENT=1 PYTHON_BIN=/tmp/heptv2-native-venv/bin/python \
bash ngt/run-model-evidence.sh migraphx BUNDLE_ROOT RESULT_ROOT/supplementary attention-decomposed-fp16
python scripts/verify-amd-attention-fusion.py RESULT_ROOT --bundle-root BUNDLE_ROOT --compress-logs --require-complete
```

Compiler tracing is enabled only for placement processes, preserving unprofiled
latency measurements. Empty disabled-pass/HIP-flag overrides select normal
attention optimization rather than HEPT's integer-sort workarounds. Keep the
actual ORT-only rocprofv3 trace and compiled programs alongside the measurements;
an environment request or a provider profile alone does not establish fusion.

At B=1,N=4000, W7900 decomposed FP16 with verified MLIR attention fusion
measured 2.54 ms and 534 MiB sampled VRAM above baseline, versus 13.36 ms
and 1419 MiB for the FP32 baseline. Maximum absolute error at this shape was
3.16e-4. The compiler and actual kernel traces confirm fused masked attention
without materialized quadratic score buffers.

## Trained-checkpoint evidence

Artifacts are separate from the seeded study, under
`onnx_validation/trained-checkpoint-evidence/2026-10-20/`. Attention uses
`attention_seed12345_20261019_162509_766691/checkpoints/checkpoint-100000.pth`;
HEPTv2 uses `heptv2_seed12345_20261020_094300_545202/checkpoints/checkpoint-20000.pth`
from `experiments/cms_backbone_comparison/`. Architectures come from their saved
`model_kwargs.pkl`; HEPTv2 width is 256. Load model weights strictly, omitting
only named training-only `task_loss_weighter.*` buffers. The independent
reference uses CPU FP32 MATH attention. Checkpoint/config hashes and exact
checkpoint inputs are retained. These checkpoints have different training
durations and are not a matched physics comparison.

The exporter requires an elementwise CMS experiment, its saved configuration, and compatible TFDS inputs. Pass the dataset versions explicitly:

```bash
uv run --project envs/ort-cpu python3 scripts/export-checkpoint-onnx.py \
  --experiment /path/to/cms/experiment \
  --bundle onnx_validation/trained-checkpoint-evidence/my-campaign/bundles/attention \
  --data-dir /path/to/cms/tfds \
  --datasets cms_pf_ttbar/1:3.0.0 cms_pf_qcd/1:3.0.0 cms_pf_ztt/1:3.0.0
```

Success writes `model.onnx`, a checksummed `manifest.json`, and reference `.npz` cases. Choose datasets that match the checkpoint's feature schema; this example reproduces the retained study's input selection. Those checkpoints were trained on CMS 2.8.0 and qualified with CMS 3.0.0 test events. The default exporter takes two actual test events each from ttbar, QCD and Ztt plus a mixed three-event batch. It removes
zero element-type rows, uses the saved feature width and sorting configuration,
and pads without truncating or repeating inputs. The seven cases span
N=3100–4900; the mixed batch has B=3,N=4900. Length and event content both vary,
so these are observed scaling measurements rather than a controlled length
sweep. Dataset, split, index and metadata hashes are recorded per case. The inputs
use raw TFDS X features; the training loader's CMS HO pT correction and target
transformations are not applied. This scope validates numerical export
equivalence, not the full training/evaluation preprocessing pipeline.

Export attention again with `--attention-format attention24 --attention-dtype
float16` for NVIDIA and `--attention-format decomposed --attention-dtype
float16` for AMD. Both optimized profiles use the same FP32 reference arrays.
The trained attention graph has three attention operations; derive fusion
counts from the graph rather than assuming the seeded model's five. Keep
`atol=1e-3, rtol=3e-3` unchanged and report violating elements and binary/PID
argmax changes. A successfully dispatched fused kernel does not establish
FP32 numerical equivalence. `scripts/diagnose-attention-precision.py` compares
the ONNX outputs with PyTorch evaluations of the same mixed-precision export
operations to help separate precision drift from additional runtime differences.

HEPTv2 export inverts permutations with exact integer ScatterElements, avoiding a flattened TopK inversion. The inverse-permutation regression includes sizes 1, 4900, and 14700. Requalify regenerated graphs on every target provider and retain their hashes.

Use the uv environments documented in `ngt/README.md`. Run the same timing,
placement, hashing, static-model retention and kernel-trace protocol as above.
The HTML report and CSV retain numerical failures explicitly. The strict
`--require-complete` gate deliberately exits nonzero when any profile fails,
even if every measurement has been collected.

All 56 trained measurements were collected. The portable FP32 baseline passes
42/42 cases, including the W7900 mixed batch. Both fused FP16
profiles execute entirely on their requested GPU and dispatch the expected
fused kernels, but fail the unchanged FP32 tolerance in all seven cases each
(0/14 accepted). Across these FP16 cases, maximum absolute error is about
0.039; binary argmax decisions change for 1–5 valid inputs per case, while PID
argmax decisions are unchanged. These optimized profiles remain unqualified
under the stated numerical criterion. The strict gate returns exit code 1,
which is retained alongside the complete report.

### FP16 fusion precision control

`scripts/compare-attention-fusion.py PACKAGE_ROOT --provider cuda|migraphx
--outdir OUTPUT` runs each retained case twice in fresh processes, using the
same FP16 ONNX graph, weights, inputs and FP32 references. This is a precision
diagnostic, not an additional timing benchmark. CUDA selects `sdpa_kernel=1`
(FlashAttention) versus `sdpa_kernel=16` (MATH). MIGraphX keeps MLIR enabled
and switches `MIGRAPHX_MLIR_USE_SPECIFIC_OPS` from `attention` to `!attention`.
Disabling all MLIR would also change the surrounding projection GEMMs and is
therefore not the attention-only control. Use the same documented runtime
libraries for both sides.

The runner verifies identical specialized graph hashes, full GPU placement,
all three CUDA attention dispatches, and AMD final compiler programs with
three fused attention objects versus zero and materialized quadratic score
buffers. It retains outputs, per-head error metrics, tolerance violations,
binary/PID argmax disagreements and execution evidence. Compare fused FP16
against FP32, unfused FP16 against FP32, and fused FP16 against unfused FP16;
the first two failing does not by itself establish that fusion adds no error.
Representative Nsight/rocprofv3 traces supplement per-case execution evidence.
Results are retained separately under the package's `fusion-precision/`.

All seven pairs per GPU were collected and their execution paths verified.
The maximum absolute errors across cases and output heads are:

| Comparison | NVIDIA L40S | AMD W7900 |
|---|---:|---:|
| Fused FP16 vs FP32 | 0.03785 | 0.03887 |
| Unfused FP16 vs FP32 | 0.03916 | 0.09872 |
| Fused FP16 vs unfused FP16 | 0.01879 | 0.08217 |

Every comparison fails the unchanged tolerance in every case (0/7 for each
row and GPU). Thus FP16 precision failure also occurs without attention
fusion, but fusion-dependent numerical differences are not negligible under
this criterion. On AMD, fused attention is closer to FP32 by maximum absolute
error in all seven cases. Binary argmax differences between fused and unfused
total 5 on NVIDIA and 32 on AMD across the seven cases; PID decisions agree.
These are casewise totals, including repeated events in the mixed batch.
All outputs are finite and padded outputs remain exactly zero. The evidence
does not support attributing every deviation exclusively to FP16 input casts
or declaring the fused and unfused implementations numerically equivalent.

Nsight confirms FlashAttention versus `UnfusedSoftmaxKernel` with separate
GEMMs. AMD rocprofv3 traces confirm fused attention kernels versus separate
GEMM and normalization kernels; these traces include compilation/tuning as
well as inference, so dispatch counts are not inference operation counts.
Per-case final programs establish the three attention operations and retained
MLIR GEMMs. The `fusion-precision/report.html` appendix links the aggregate and
per-case metrics, with independently checksummed outputs and execution evidence.
