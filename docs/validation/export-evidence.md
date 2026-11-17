# Full-model export evidence protocol

Use this protocol to qualify an elementwise attention or HEPTv2 export on a specific runtime. Start with the [ONNX guide](onnx.md) for the input contract and a CPU check. Qualification requires numerical agreement, provider-placement evidence, and reproducible measurements. A kernel dispatch or successful export alone is insufficient.

## Current qualification boundary

| Profile | Retained evidence | Interpretation |
|---|---|---|
| Portable FP32 attention and HEPTv2 | 42/42 trained-checkpoint cases pass across CPU, NVIDIA L40S, and AMD W7900. | Qualified for the recorded checkpoints, inputs, and runtime stacks. |
| CUDA Attention-24 with FP16 attention | GPU placement and fused kernels verified; 0/7 trained cases pass the FP32 tolerance. | Executable, but not numerically qualified under that criterion. |
| AMD decomposed FP16 attention with MLIR fusion | GPU placement and fused kernels verified; 0/7 trained cases pass the FP32 tolerance. | Executable, but not numerically qualified under that criterion. |
| Full-model BF16 CMS v3.2.0 | All eight ONNX scenarios complete the 5,000-event test sample; all strict numerical gates fail. | Physics distributions and runtime are reported separately; no numerical-equivalence claim. |
| Particle-set output | No export contract in `export_mlpf`. | Requires separate implementation and validation. |

The acceptance tolerance in these studies is `atol=1e-3, rtol=3e-3`. Evidence is artifact-specific; it does not establish reconstruction physics quality or qualify all model shapes and environments. Detailed measurements and provenance live in the retained study packages indexed below.

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

## Controlled-shape protocol

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

The historical controlled-sweep models used seed 42 and untrained weights. Both had three backbone
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

## CMS checkpoint and physics qualification

The CMS v3.2.0 studies use release revision
`0acee6877912fd61fb7d6f03bf90348d33b441f4` of `jpata/particleflow`, with both
checkpoints at step 100000. Verify the published hashes in `cms/v3.2.0/release.json`.
These checkpoints differ from the earlier trained-checkpoint shape study.

`scripts/qualify-cms-physics.py` exposes `prepare`, `export`, `reference` and `run`.
Each takes `--root PACKAGE_ROOT`; model operations select `--model attention|heptv2`.
The package needs `release.json` and matching checkpoints/hyperparameters under
`checkpoints/MODEL/`. Prepare once from `cms_pf_ttbar/1/2.8.0` with
`--data-dir TFDS_ROOT`. Use `--events 5000` for the full test sample or label a
smaller smoke sample explicitly. Preparation applies production `TFDSDataSource`
preprocessing and saved sorting, rejects duplicate serialized records/event
content, and retains hashes and bucket sizes. Every runtime consumes the same
prepared inputs.

Export/runtime profiles are `fp32`, `decomposed-fp16`, `attention24-fp16`,
`decomposed-bf16` and `attention24-bf16`; execution selects
`--provider cpu|cuda|migraphx`. BF16 export needs a GPU. `reference` is independent
CPU FP32 MATH; use `scripts/qualify-cms-bf16.py --root PACKAGE_ROOT --model MODEL`
for GPU BF16 references. Keep native AMD controls separate from the common
NVIDIA BF16 reference. Each static runtime bucket runs in a fresh process.
Require finite heads and exact zero ONNX padding; compare valid eager positions.

`scripts/analyze-cms-physics-closure.py --root PACKAGE_ROOT --model MODEL
--candidate RESULT_DIRECTORY --outdir REPORT_DIRECTORY` checks hashes/placement
and invokes production particle decoding. Samples below 1,000 events require
`--allow-small-sample`. Physics remains report-only; retain the unchanged raw-head
numerical gates alongside particle/jet/MET results.

Cluster all collections with CMS anti-kT R=0.4, pT>3 GeV and FP64 four-vectors.
Retain targets/candidates, gen jets/MET, species, counts, event tails and one-to-one
jet matches at dR<0.1. PU-reduced collections use `ispu<0.8`. Preserve per-event
pairing, 400 event-bootstrap resamples, histogram underflow/overflow and sparse-bin
information. Whole-event reference sampling bands are pointwise comparisons,
not a global significance test or a physics-equivalence margin.

### Full-model BF16 and fusion controls

The BF16 policy uses GPU autocast, FP32 LayerNorm, original FP32 geometry,
exact int64 HEPT hashing and FP32 external I/O, with TF32 off. Native Attention
uses forced SDPA Flash with production jagged packing; HEPTv2 uses compiled
FlexAttention with FP32 log-sum-exp. ONNX Attention-24 exposes no LSE, so HEPTv2
retains a separate FP32 QK/LSE branch for hash-bucket merging. Its cost is included;
fusion is partial.

The export repairs retain stable max-shifted log-sum-exp, exact BF16 raw literals
and FP32 widening for unsupported BF16 predicates. The BF16 lattice audit still
found off-lattice AMD ONNX classification values, while native GPU references and
NVIDIA ONNX preserved rounding. Record that compiler discrepancy without changing
the reference/tolerance or attributing every prediction difference to rounding.

Verify full GPU placement and actual fusion separately on exact graph hashes.
CUDA selects Flash versus MATH dispatch. AMD final compiler programs must show
fused attention objects and the expected quadratic-buffer behavior. Use
`verify-cms-physics-fusion.py` for FP16 and `verify-cms-bf16-fusion.py` for BF16.
The AMD attention-only control toggles `MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention`
versus `!attention`; disabling all MLIR changes projection GEMMs too.

BF16 batch-one timing uses six warmups per static bucket and synchronous CPU
FP32 inputs to CPU FP32 outputs, including transfers/conversion. File I/O,
compilation, physics and profiling are excluded from timed inference. Report
session creation, first run and compilation separately. Keep latency samples,
tails and the actual event-size/bucket mix.

`scripts/benchmark-cms-throughput.py --root PACKAGE_ROOT --outdir OUT
--provider cuda|migraphx` measures native/ONNX batches in isolated processes.
Use `--engines native` or `--engines onnx` for separate runtime environments.
Defaults are batches 1/2/4/8/16, lengths 4000/5000, 64 distinct events per length,
six warmups and eight timed rounds. These fixed-bucket results are not the full
test-size mixture or a batched physics-equivalence qualification.
`scripts/plot-cms-throughput.py --root STUDY` renders collected result directories.

## Retained studies

These are retained local/EOS packages, not artifacts downloaded by a repository
checkout. Open `report.html` in each package for complete measurements, source,
failed attempts and provenance. Preserve the records when running a new campaign.

| Study | Artifact location | Interpretation |
| --- | --- | --- |
| Seeded sweep, 2026-10-19 | `onnx_validation/full-model-evidence/2026-10-19/` | Controlled/repeated features; not trained physics |
| Trained checkpoints, 2026-10-20 | `onnx_validation/trained-checkpoint-evidence/2026-10-20/` | 42/42 portable FP32 cases pass; 0/7 trained FP16 cases pass per GPU; fusion/precision controls retained separately |
| CMS v3.2.0, 2026-10-30 | `onnx_validation/cms-v3.2.0-qualification/2026-10-30/` | 1,024 unique events; original AMD zero-output failure and stable-LSE repair retained separately |
| BF16 baseline, 2026-10-30 | `onnx_validation/cms-v3.2.0-bf16-flash/2026-10-30/` | 1,024 events with full-model BF16 policy and platform controls |
| Expanded BF16, 2026-11-02 | `onnx_validation/cms-v3.2.0-bf16-flash/2026-11-02-expanded/` | All 5,000 test events, eight ONNX scenarios, 14 physics comparisons, 40 fusion checks; strict numerical gates fail |
| Standalone physics/latency | `/eos/user/j/jpata/mlpf_studies/20261102_cms_onnx_physics_latency/` | `render.sh` rebuilds plots/reports from copied histogram/timing inputs |
| Batch throughput, 2026-11-03 | `/eos/user/j/jpata/mlpf_studies/20261103_cms_batch_throughput/` | 80 points; CUDA dispatch verified for all 20 timed ONNX graphs; AMD fusion requested but not independently requalified per batch shape |

The expanded report retains AMD particle-count shifts and low-pT spectral
departures. Small total count changes or spectra inside reference sampling bands
do not override the failed raw-head gates. Historical reports describe their
recorded stacks; their case counts, timings and conclusions should not be
hardcoded into current regression tests.
