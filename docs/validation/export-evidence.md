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

## CMS v3.2.0 numerical and physics closure

The 2026-10-30 rerun uses main revision
`4e0da237661fcdea5f7dc3e18b45142486833a1c` and Hugging Face release revision
`0acee6877912fd61fb7d6f03bf90348d33b441f4` of `jpata/particleflow`.
Both v3.2.0 checkpoints are at step 100000. Verify their published SHA256
values from the pinned `cms/v3.2.0/release.json`; these are different models
from the 2026-10-20 qualification.

Artifacts belong under `onnx_validation/cms-v3.2.0-qualification/2026-10-30/`.
The input manifest selects 1024 unique test events with seed 20261030 from
`/eos/user/j/jpata/mlpf/cms/2.8.0/cms_pf_ttbar/1/2.8.0`. All 64 test shards
are represented; valid lengths span 2018–5516. Both serialized-record and
event-content hashes are checked for duplicates. No repeated event contributes
to the 1024-event requirement. The production `TFDSDataSource` applies the CMS
HO correction, target remapping, feature-width handling and saved sorting
settings. Four static length buckets (3000, 4000, 5000, 6000) pad without
truncation. Every model/runtime consumes the same prepared inputs and masks.

`scripts/qualify-cms-physics.py` provides `prepare`, `export`, `reference` and
`run` commands. Each requires `--root PACKAGE_ROOT`; export/reference/run
select `--model attention|heptv2`. Attention also supports
`--profile decomposed-fp16|attention24-fp16`. Runtime selection is
`--provider cpu|cuda|migraphx`, with `--fusion on|off` for the FP16 controls.
CPU references use independent PyTorch FP32 MATH evaluation. Eager padded
logits need not vanish; retain valid positions for comparison and enforce
exact zero padding on ONNX outputs. Static runtime shapes execute in separate
processes with full-GPU placement checks and retained graphs/profiles.

`scripts/analyze-cms-physics-closure.py --root PACKAGE_ROOT --model MODEL
--candidate RESULT_DIRECTORY --outdir REPORT_DIRECTORY` verifies input/output
hashes and runtime placement before comparing numerical heads. It invokes
the production `MLPF.predict_particles` decoding on retained heads, avoiding
a separate implementation of particle-presence/PID selection or momentum
rescaling. Particle and jet collections, per-event distributions and paired
event-bootstrap intervals are retained. Physics is report-only by explicit
user instruction; numerical acceptance remains `atol=1e-3, rtol=3e-3`.
Reports below 1000 events require `--allow-small-sample` and are labeled as
smoke tests. Never present a partially collected matrix as complete.

Cluster every decoded collection using the current CMS anti-kT R=0.4 and
pT>3 GeV selection, with the same FastJet implementation and FP64 four-vector
construction. Include targets/candidates as context, and preserve gen jets and
MET. Compare particle kinematics, species, multiplicities and PID migrations;
jet kinematics, multiplicities, leading/subleading jets and one-to-one angular
matches (dR<0.1); and event MET/summed pT. The PU-reduced collection uses the
production `ispu<0.8` selection. Histogram differences use 400 paired
event-bootstrap resamples; zero-reference bins are omitted from ratio plots
and underflow/overflow counts remain in the numerical histogram artifacts.
Distribution agreement does not override raw-head numerical failures.

The 1024-event GPU rerun completed on NVIDIA L40S (CUDA ORT 1.29) and AMD
W7900 (MIGraphX ORT 1.23.2 / MIGraphX 2.18). Attention FP32 passes on both:
maximum absolute head errors are 0.0003681 and 0.0003772 respectively.
HEPTv2 FP32 passes on NVIDIA (0.0002203), but AMD returns zero heads and
fails (20.85485). The AMD failure reproduces with the documented compiler
recipe configured before loading ORT and with a no-MLIR/rocBLAS control.
A standalone ONNX/input/expected-output reproducer is retained; this model
remains unqualified on this AMD runtime; the subsequent repair is documented below.

Attention FP16 fails the unchanged numerical tolerance with fusion enabled
and disabled on both GPUs. Maximum errors against FP32 are 0.11786/0.12123
on NVIDIA and 0.11893/0.47344 on AMD (fused/unfused). Direct fused-versus-
unfused errors are 0.08685 and 0.42663. Thus failure also occurs without
fusion, but fusion contributes measurable differences and cannot be excluded.
Per-shape programs and actual Nsight/rocprof traces establish the selected
attention implementation; compilation/tuning dispatches are not counted as
inference operation counts.

Physics differences are reported without assigning acceptance. Across 1024
events, FP32 Attention changes net particle counts by -1/-3 on NVIDIA/AMD
and net jets above 30 GeV by +6/-4. NVIDIA HEPTv2 retains particle decisions
but changes the net hard-jet count by +9. Jet matching tails and individual
event outliers remain visible despite small average shifts. AMD HEPTv2
reconstructs no particles or jets, versus 3,145,016 particles and 28,005 jets
above 30 GeV in its reference. Consult the aggregate HTML and each model's
plots, paired-event parquet, histogram arrays and confidence intervals.

The standalone AMD reproducer was executed twice per session on MIGraphX
and on the CPU backend: AMD returns zero bid/id/momentum on both calls,
while CPU produces nonzero heads and passes the same FP32 tolerance. Logs
record the actual loaded MIGraphX 2.18 libraries. An eight-event physics
self-comparison smoke control gives zero shifts, unit matched-jet response
and no unmatched jets; it does not contribute to the 1024-event requirement.

Supplementary CPU-only ORT 1.29 controls complete all 1024 events. Attention
passes with default graph optimization (max error 0.0004892). Default HEPTv2
CPU execution and the eight-thread basic N=5000 run terminate with SIGBUS;
retain these as failed controls. HEPTv2 passes with basic optimization and
memory patterns disabled (max error 0.0005445), using eight threads for
N=3000/4000 and one for N=5000/6000. This mixed configuration is a correctness
control and supports no CPU performance comparison. The initial rerun collected
12 complete analyses: six FP32 runtime controls, four FP16 profiles and two
fused-versus-unfused comparisons. Its strict numerical gate fails, as expected.

### AMD HEPTv2 zero-output repair

The follow-up isolated `ReduceLogSumExp` in MIGraphX 2.18. Its ONNX parser
uses `log(sum(exp(x)))`, without a maximum shift. A tiny operator reproducer
returns +inf for `[100,101]` and -inf for `[-1000,-1001]` on AMD, while CPU
and max-shifted arithmetic give the correct finite results. In the released
checkpoint's second attention layer, all 192,000 CPU log-sum-exp values are
finite (maximum 170.160736), while 41,668 AMD values are +inf. Extracting the
binary head before cleanup gives 8,000/8,000 NaN AMD logits; the model's final
`nan_to_num` turns these into zeros, hiding the internal nonfinite values.

`mlpf/model/onnx_graph.py` now lowers log-sum-exp to explicit max-shifted
primitives before export cleanup. A zero shift for infinite maxima preserves
infinity semantics; axes, keepdims and empty-axis/noop behavior are retained.
Eager evaluation, independent reference outputs and checkpoint weights do
not change. Six new regression cases cover overflow, underflow, infinities,
NaNs, opsets 17/20, keepdims and empty axes. All 26 targeted CPU tests and
all six reduction cases directly on AMD (CPU fallback disabled) pass.

A fresh export completes the same 1024-event sample on AMD with zero numerical
violations and max absolute head error 0.00020980835. Particle multiplicities
match in every event. Jet count differences are -9 above 3 GeV and -7 above
30 GeV; mean MET shift is 6.46e-8 GeV, paired 95% CI [-6.64e-6, 6.83e-6].
Physics acceptance remains unassigned. The aggregate report retains the old
failure and adds a separately labeled repaired-export row (13 analyses).
Debug evidence, operator/layer reproducers, executed source, fresh graphs,
raw outputs and the physics report are under `heptv2-amd-debug/` and
`analysis/heptv2-amd-stable-logsumexp/` in the qualification package.

### CMS BF16 Flash baseline qualification

The separate rerun is retained at
`onnx_validation/cms-v3.2.0-bf16-flash/2026-10-30/`. It reuses the same pinned
CMS v3.2.0 step-100000 checkpoints and 1024 unique CMS 2.8.0 ttbar events.
The common numerical and physics reference is NVIDIA L40S PyTorch BF16:
forced SDPA Flash with production jagged packing for Attention, and compiled
FlexAttention with FP32 log-sum-exp for HEPTv2. Native AMD W7900 PyTorch BF16
references are retained as separate platform controls and timing baselines.
HEPTv2 is explicitly labeled FlexAttention, since its attention must return
log-sum-exp for merging hash buckets.

The policy is full-model GPU BF16 autocast, FP32 LayerNorm, original FP32
geometry and exact int64 HEPT hashing, and FP32 external inputs/outputs.
This differs from the earlier attention-only FP16 cast profiles. TF32 is off.
The reference keeps its autocast weight cache across events. All timings are
batch-one synchronous CPU inputs to CPU FP32 outputs, including transfers
and output conversion, after six warmups per static bucket. File I/O,
compilation, physics and profiling are excluded from measured inference.
Cold session creation, first run, compilation warmup and per-bucket results
are reported separately. GPU workloads run serially on each device.

NVIDIA uses standard ONNX Attention-24 BF16 with explicit Flash/MATH dispatch.
AMD uses decomposed BF16 attention with MIGraphX MLIR attention enabled or
disabled, retaining MLIR for other operations. Fusion is checked separately
from provider placement. HEPTv2's standard ONNX fused export retains a
separate FP32 QK/log-sum-exp branch because Attention-24 has no LSE output;
its measured runtime includes this extra work. It is a partial fusion of
the native FlexAttention operation.

Portability changes required by this rerun:

- Explicit boolean mask casts avoid legacy export's invalid BF16 promotion
  to ONNX COMPLEX128 while preserving eager arithmetic.
- BF16 ELU uses FP32 opmath and one BF16 rounding. BF16 comparison operands
  widen exactly to FP32 because CUDA ORT 1.29 lacks those predicate kernels.
- BF16 literals use exact raw bytes: MIGraphX accepts BF16 raw tensors but
  cannot parse BF16 literals serialized using `int32_data`.
- HEPTv2 requests LSE through `AuxRequest` on newer PyTorch, avoiding a
  deprecated `return_lse` warning that breaks full-graph compilation.
  Older PyTorch retains the previous API. Dense GPU fallback stays disabled.

These changes preserve checkpoints and keep the earlier stable log-sum-exp
repair. The final targeted suite passes 49 tests, covering Attention-24,
HEPTv2 sorting/geometry, stable reductions, BF16 graph structure, exact
predicate widening, raw BF16 bit preservation and physics helpers. Both
native GPU references also complete all 1024 events using their required
fused kernels. Physics acceptance stays report-only, with no inferred limit;
raw-head tolerance stays `atol=1e-3, rtol=3e-3`.

The BF16 labels describe the exported graph and verified attention kernels,
not guaranteed identical intermediate rounding across compilers. An audit
of the classification heads checks that values lie on the BF16 lattice
before the external FP32 cast. NVIDIA preserves this rounding; AMD emits
off-lattice values in both models, including HEPTv2 with
`eliminate_data_type` disabled. Final compiler programs show BF16 matrix
outputs fused with FP32 bias/cleanup arithmetic. This backend discrepancy
is retained and reported; the baseline and tolerance are not adjusted to
hide it. `bf16-rounding-audit.json` and the operator probes retain the
precision evidence separately from the attention-fusion evidence.

All eight ONNX scenarios complete all 1024 events, with full GPU placement,
finite heads and zero padded outputs. All 32 model/device/mode/shape fusion
checks pass. All eight raw-head numerical gates fail. The separate report
contains 14 complete physics analyses: eight ONNX-versus-reference cases,
four same-device fused-versus-unfused comparisons, and two native AMD
PyTorch-versus-NVIDIA controls.

The table lists fused / unfused ONNX results. Count changes are totals over
1024 events relative to the common NVIDIA PyTorch BF16 reference.

| Model / GPU | Native PyTorch mean ms | ONNX mean ms F / U | Max abs error F / U | Δ particles F / U | Δ jets ≥30 GeV F / U |
|---|---:|---:|---:|---:|---:|
| Attention / L40S | 13.23 | 4.18 / 20.96 | 1.0625 / 1.2969 | -15 / -25 | -12 / +6 |
| Attention / W7900 | 15.72 | 8.54 / 18.55 | 1.3497 / 4.3497 | +2770 / +1850 | +2 / +11 |
| HEPTv2 / L40S | 17.13 | 25.72 / 27.09 | 1.9375 / 2.3125 | -105 / -70 | +3 / -52 |
| HEPTv2 / W7900 | 20.84 | 29.36 / 30.41 | 2.1982 / 3.2720 | +4204 / +4511 | +5 / -35 |

Attention ONNX fusion improves steady-state performance; native PyTorch
remains faster for HEPTv2 on both devices. AMD HEPTv2 fused session creation
costs 50.5–256.9 seconds per static shape, excluded from inference timings.
The final source regenerates all four measured BF16 exports byte-for-byte.

Small aggregate jet/MET differences do not exclude particle biases or rare
event changes. Fused AMD Attention adds 2.705 particles/event (95% CI
[2.538, 2.892]) and fused AMD HEPTv2 adds 4.105 ([3.948, 4.286]). Their mean
scalar-sum-pT shifts are +2.254 and +2.651 GeV/event. Fused AMD HEPTv2 mean
MET shifts +0.191 GeV, CI [0.0066, 0.3621]; the maximum event shift is 47.19
GeV. These are measured differences, with no physics acceptance assigned.

| Direct fused minus unfused BF16 | Max abs raw error | Δ particles | Δ jets ≥30 GeV | Max event abs(ΔMET), GeV |
|---|---:|---:|---:|---:|
| Attention / L40S | 0.96875 | +10 | -18 | 52.48 |
| Attention / W7900 | 3.28125 | +920 | -9 | 35.58 |
| HEPTv2 / L40S | 1.34375 | -35 | +55 | 64.94 |
| HEPTv2 / W7900 | 2.71875 | -307 | +40 | 66.64 |

The large tails include high-pT presence decisions: event_0520 gains a
53.9 GeV electron in fused NVIDIA Attention, and event_0239 gains a
77.2 GeV electron in fused NVIDIA HEPTv2 relative to their unfused outputs.
Per-comparison `met-outliers.json` records the affected elements and decoded
four-vectors. Both native AMD PyTorch controls also fail the strict raw-head
gate against NVIDIA, so all export error cannot be attributed to attention
fusion alone.

The AMD operator reproducer confirms a separate rounding discrepancy without
attention: BF16(1 + BF16(0.003)), widened to FP32, should return 1.0 but
returns 1.0030059814453125, with or without `eliminate_data_type` disabled.
The whole-sample audit finds off-BF16-lattice logits on AMD, while both
native PyTorch baselines and NVIDIA ONNX preserve BF16 classification-head
rounding. BF16 alone therefore does not establish backend compatibility.
The exact compiler/runtime builds, raw arrays, kernel/compiler proofs,
paired physics arrays, plots, failed attempts and source snapshots are
retained in the BF16 package; the previous FP32/FP16 report is unchanged.


## Expanded BF16 qualification: all ttbar test events (2026-11-02)

The expanded package is retained at
`onnx_validation/cms-v3.2.0-bf16-flash/2026-11-02-expanded/`
(a workspace link to EOS). Open `report.html` and `relative-counts/report.html`.
The requested 10,000 events were revised to all 5,000 available ttbar test
events; the original 1,024 inputs are preserved exactly as a prefix. Checkpoints,
graphs, precision and runtime profiles are unchanged. All eight ONNX scenarios,
four native reference scenarios, 14 physics comparisons and 40 fusion checks
complete on this expanded sample.

Count shifts below are percentages of the common NVIDIA PyTorch BF16 reference
count, using total counts across all 5,000 events. F / U means fused / unfused.

| Model / GPU | Δ particles [%] F / U | Δ jets ≥30 GeV [%] F / U |
|---|---:|---:|
| Attention / L40S | -0.00027 / +0.00003 | -0.09388 / -0.17829 |
| Attention / W7900 | +0.08735 / +0.05772 | -0.04891 / -0.12623 |
| HEPTv2 / L40S | -0.00526 / -0.00424 | +0.05787 / -0.00519 |
| HEPTv2 / W7900 | +0.13242 / +0.14489 | -0.09570 / -0.15951 |

The AMD excess in particle counts persists with increased statistics and is
present with fusion disabled. The full-sample BF16 lattice audit again finds
off-lattice classification heads in AMD ONNX, while both native references and
NVIDIA ONNX preserve BF16 rounding. Fusion still changes some predictions: the
direct fused-minus-unfused AMD particle shifts are +0.02962% (Attention) and
−0.01245% (HEPTv2), relative to the unfused outputs. These observations do not
attribute every difference to rounding.

All eight strict ONNX numerical gates still fail; direct fusion comparisons and
native GPU controls also fail the unchanged raw-head tolerance. Physics remains
report-only. The distribution pages retain event tails, gains/losses and matching
results rather than interpreting small net count shifts as numerical equivalence.
The NVIDIA SIGBUS attempt is retained alongside its successful complete-bucket
retry. Full predictions and paired physics arrays are archived on EOS.

The expanded report also embeds latency distributions and latency versus valid
event size for all 12 runtime scenarios, using the original 60,000 timed records.
`runtime-plots/` contains PNG/PDF figures, per-event CSV, source hashes and
validation results. Static bucket boundaries and 250-element median/percentile
summaries distinguish ONNX shape steps from the native size dependence.

The current main report prioritizes particle-pT, jet-pT and event-sum-pT spectra
with event-level reference sampling bands and paired bootstrap difference
intervals. Raw numerical gates are retained under technical diagnostics. Among
fused scenarios, populated jet-pT and sum-pT bins stay within the reference ±2σ
sampling band; AMD HEPTv2 has four particle-pT bins outside it, all below 0.63 GeV
(maximum 4.88σ at 0.15–0.18 GeV). This is a pointwise finite-sample comparison,
not a global significance test or a physics equivalence margin. Latency plots
and the runtime comparison remain in the main report.

A standalone copy is available at
`/eos/user/j/jpata/mlpf_studies/20261102_cms_onnx_physics_latency/`.
Open `report.html`; `render.sh` regenerates the report and figures using only
the copied histogram and timing inputs. All 472 local report links/images were
verified after reproduction; the copy has no symlinks or source-directory dependency.

### CMS BF16 batch throughput

The completed 2026-11-03 sweep is retained at
`/eos/user/j/jpata/mlpf_studies/20261103_cms_batch_throughput/`.
Open `report.html` for throughput and batch-latency plots and `throughput.csv`
for all 80 points. It compares native PyTorch and fusion-enabled ONNX on both
GPUs, for Attention and HEPTv2, batches 1/2/4/8/16 and static lengths 4000/5000.
Each length uses the same 64 distinct real events across scenarios, repeated
eight times after six warmups in a fresh process per shape. Timings include
synchronous CPU FP32 inputs to CPU FP32 outputs and transfers; input preparation,
compilation, warmup, profiling and output validation are excluded. Throughput is
total events divided by summed inference time, excluding gaps between calls.

At 4000 slots, L40S ONNX Attention peaks at batch 2 (281 events/s versus 257 at
batch 1), while native Attention peaks at batch 8 (295 versus 82 events/s).
L40S ONNX HEPTv2 also peaks at batch 2 (59 versus 53 events/s). Larger batches
reduce L40S ONNX throughput at both lengths. W7900 ONNX generally benefits
through batches 8–16; at 5000 slots, HEPTv2 improves from 16.2 to 45.7 events/s.
These fixed-bucket results do not represent the full 5000-event size mixture.

All 80 points completed with finite outputs and verified ONNX GPU placement.
CUDA Flash dispatch was checked separately for all 20 exact timed ONNX graphs;
AMD fusion was requested but not independently requalified for every batch shape.
First-batch raw-head comparisons against retained batch-one predictions show
differences, so this is not a batched physics-equivalence qualification.
Removing the first timed round changes throughput by at most 0.635%.

`scripts/benchmark-cms-throughput.py` runs isolated shape processes against the
original evidence root (with inputs, checkpoints and prediction baselines).
Use `--provider cuda` or `--provider migraphx`, `--root ROOT`, and `--outdir OUT`;
`--engines onnx` and `--engines native` allow separate runtime environments.
Use the qualified GPU environments described in `ngt/README.md`.
`scripts/plot-cms-throughput.py --root STUDY` renders collected result directories
without GPU access. The standalone study retains executed source archives,
launch commands, logs, numerical diagnostics, raw timings and checksums.
