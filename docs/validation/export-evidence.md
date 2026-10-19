# Full-model export evidence protocol

The study covers the attention and deterministic-hashing HEPTv2 full MLPF
models using the new export path. The available NVIDIA machine is **L40S**,
not L40; the AMD machine is W7900. Both existing NGT pods have a ten-hour
lifetime. Preserve results locally before expiry.

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

The initial models use seed 42 and untrained weights. Both have three backbone
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
hashes, model hashes and input hashes. AMD uses the previously qualified native
ORT 1.23.2 with MIGraphX 2.18/ROCm SDK libraries and the documented HEPT integer
and compiler workarounds. This is a comparison of complete software stacks.

## Reproduction and artifacts

`scripts/qualify-heptv2-onnx.py export --model-type attention|heptv2` generates
bundles; `--inputs-bundle` reuses CMS inputs, and `--sweep` plus `--batches`
generates the controlled shapes. `scripts/benchmark-model-export.py` measures
one case. `ngt/run-model-evidence.sh PROVIDER BUNDLE_ROOT RESULT_ROOT` runs the
matrix inside a configured runtime; see ngt/README.md for NVIDIA image and AMD
library setup. It bounds each worker to 30 minutes and retains each exit code.

Local artifacts belong under `onnx_validation/full-model-evidence/2026-10-19/`:
model/reference bundles, raw results and memory samples, placement profiles,
logs, machine/source provenance, aggregate tables and scaling plots. The path
is ignored by Git; retain source scripts and this protocol in Git separately.
Run `scripts/summarize-model-evidence.py PACKAGE_ROOT --plots --require-complete`
and `scripts/render-model-evidence.py PACKAGE_ROOT` to build the final table,
CSV, PNG/PDF/SVG figures and portable HTML report. The acceptance gate includes
the supplementary profiles when their manifests are present (56 total cases).

## Initial measurements

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
