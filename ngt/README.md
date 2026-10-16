# Interactive W7900 SDPA benchmark

The legacy benchmark commands and `run-*-w7900.sh` helpers require the
benchmark changes from PR #523 (commit `315c1b7e`). This main-based study
branch instead provides its standalone runner under `scripts/attention-study/`;
see `docs/attention-study/reproduce.md`.

## Start the session

Set `CERN_USER` in `ngt/.env` (created from `.env.example`, ignored by Git).
The namespace must be your CERN username. Access requires membership in
`ngt-access-all` and an initial login to https://ngt.cern.ch.

The Kubernetes API is not directly accessible from outside CERN. Connect to
[CERN eduVPN](https://eduvpn.docs.cern.ch/) before running the launcher, or run
it from lxplus. A timeout reaching `188.185.128.201:6443` indicates a network
connectivity problem; reinstalling the OIDC plugin will not resolve it. See
[NGT remote access](https://gitlab.cern.ch/mlops/ngt/docs/-/blob/master/docs/getting-started.md#remote-access) for tunnel alternatives.

The launcher uses `ngt/.kube/config`, leaving your existing kubeconfig alone.
Install `kubectl` and `kubectl oidc-login` v1.32.2 as described in
[the NGT docs](https://gitlab.cern.ch/mlops/ngt/docs/-/blob/master/docs/getting-started.md); both are available on lxplus.
The launcher also searches `ngt/.bin/` for tools. On this workstation,
kubelogin v1.32.2 is installed there as `kubectl-oidc_login` (ignored by Git).
On this workstation the snap wrapper cannot run in the agent sandbox; the
underlying executable works, so you can set
`KUBECTL=/snap/kubectl/current/kubectl` in `.env`.

From the repository root:

```bash
bash ngt/session.sh init      # CERN SSO authentication
bash ngt/session.sh create    # one W7900, 10-hour lifetime
bash ngt/session.sh status    # repeat until Running
bash ngt/session.sh copy      # copies benchmark/setup to /tmp/particleflow
bash ngt/session.sh shell
```

If the pod stays Pending, inspect it with the same explicit kubeconfig,
context and namespace; GPU capacity may be exhausted. A shell exit leaves the
session running. Stop it explicitly when finished.

## Configure the environment inside the pod

The checked-in NGT docs report ROCm 7.2.4 (and 6.4.3 as an alternative).
`envs/ort-amd` targets a different ROCm 10.1 plugin stack and lacks PyTorch;
do not use it for this setup. The recipe below uses AMD's published ROCm
7.2.1 wheels: PyTorch 2.9.1 and ONNX Runtime MIGraphX 1.23.2, Python 3.12.
Compatibility with the actual NGT image's ROCm/MIGraphX libraries must be
confirmed by the smoke run. The existing W7900 pod was tested on 2026-10-16
with ROCm/MIGraphX 7.2.4, PyTorch 2.9.1 (ROCm 7.2.1), and ORT 1.23.2.
Its ORT wheel also needs a newer C++ runtime than the Alma9 default; the pod
already has one in `/tmp/sdpa-cxx-runtime/lib`. Prepend that directory to
`LD_LIBRARY_PATH` when using this environment.

```bash
ls -d /opt/rocm*
rpm -qa | grep -Ei 'migraphx|rocm|half'
# Use the actual installed 7.2.x path if /opt/rocm points elsewhere:
export ROCM_PATH=/opt/rocm
sudo dnf install -y python3.12 python3.12-pip
cd /tmp/particleflow
bash ngt/setup-w7900.sh
```

MIGraphX and its ROCm shared libraries must be installed in the container as
well as the Python wheel. If absent, check the image's configured package
repositories (`dnf list --available '*migraphx*'`) and install the MIGraphX
package matching its ROCm 7.2.x installation. Do not install a different host
GPU driver. A missing `libmigraphx*.so` error at session creation means this
native prerequisite is not yet satisfied.

The setup script creates `/tmp/sdpa-w7900-venv`, checks the gfx1100 device,
installs packages and performs a PyTorch GPU matmul. Follow its printed
activation commands (or the following defaults):

```bash
export LD_LIBRARY_PATH="/tmp/sdpa-cxx-runtime/lib:$ROCM_PATH/lib:$ROCM_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source /tmp/sdpa-w7900-venv/bin/activate
python scripts/benchmark-sdpa-onnx.py \
  --device cuda --ort-provider migraphx --require-full-gpu \
  --variants pt_math_fp32 onnx_sdpa_export_fp32 \
  --export-shapes static --seq-lens 128 512 --modes both \
  --warmup 2 --iters 5 --profile --outdir results/smoke
```

PyTorch calls AMD devices `cuda` too. `--require-full-gpu` disables CPU node
fallback so unsupported graphs fail instead of producing misleading GPU
measurements. Check every row in `summary.json`: this benchmark catches
per-variant failures and may exit successfully even when rows fail. Provider
listing alone does not prove the native provider can load or run a model.

The tested stack runs the unmasked static FP32 export on MIGraphX. The masked
export fails with CPU fallback disabled because some nodes are assigned to
the CPU provider. Keep that failure visible rather than reporting it as a
full GPU measurement.

After the smoke rows succeed, increase `--seq-lens` to `512 1024 2048 4096`
and `--iters` to `20`, writing to a new results directory. Test
`pt_flash_fp16`, `ort_mha_fp16`, and `ort_attention24_fp16` separately; support
on W7900/MIGraphX is not guaranteed. Dynamic shapes and INT8 should likewise
be separate experiments. `sdpa_kernel` is a CUDA-provider option and must
not be passed to MIGraphX.

AMD VRAM values are left unavailable: the existing NVIDIA sampler cannot
measure MIGraphX memory, and PyTorch allocator statistics exclude it. ORT
timings retain the script's existing `session.run` semantics, including host
input/output transfers; they are not isolated GPU-kernel timings.

## A100/L40S comparison

`onnx_validation/pr523/run-static-full.sh` defines the NVIDIA reference sweeps.
Inside the configured pod, `bash ngt/run-pr523-w7900.sh` reproduces their shapes,
mask policies, FP16 variants, 5 warmups and 30 iterations using MIGraphX.
Copy this runner to the pod separately; `session.sh copy` copies only the
benchmark and setup script. CUDA-only provider options are omitted. The sequence
sweep additionally records profiles.

On the tested ORT 1.23.2 / MIGraphX 7.2.4 stack, unmasked PyTorch FlashAttention
and ORT MultiHeadAttention run. Masked FlashAttention needs the missing
`flash_attn` module; masked MultiHeadAttention fails the MIGraphX parser with
`Bias must be 1D shape`. Attention-24 is rejected because this ORT build
supports released opsets only through 23. These failures are retained in
the summaries, so the masked batch sweep has no successful rows. The reference
NVIDIA runs used newer PyTorch/ORT versions; timings compare complete software
stacks and include ORT host transfers.

## MATH export and kernel inspection

`bash ngt/run-math-w7900.sh` runs PyTorch MATH and the corresponding ONNX
exports in FP32/FP16 at batch 4 and sequence lengths 128, 512, 1280, 2560.
Copy this runner and `ngt/probe-ort-kernels.py` into the pod separately first.
The benchmark explicitly selects `SDPBackend.MATH` during export and retains
models and ORT profiles. The runner then uses `rocprofv3` and
`MIGRAPHX_TRACE_COMPILE=1` on an ORT-only probe at sequence length 512, so
PyTorch reference kernels cannot contaminate the trace.

The 2026-10-16 run passed all 16 PyTorch cases and 8 unmasked ONNX cases;
all 8 masked ONNX cases required CPU fallback and failed the strict check.
At sequence length 512, both FP32 and FP16 compiled programs allocate full
`[4,16,512,512]` attention intermediates and execute separate QK matmul,
softmax, and probability-times-V matmul kernels. Neither uses FlashAttention
in these traced cases. Aggregate profiler statistics include compiler tuning
runs; inspect the final nine dispatches (three inference iterations) and the
final compiled program for the actual inference sequence. Artifacts are in
`onnx_validation/pr523/static-rerun/w7900/math/`, with full working results
also fetched under `ngt/results/math-w7900/`.

The masked MATH export diagnosis identified `/IsNaN` as the sole CPU node
in the profiled FP16 S=512 graph. PyTorch exports safe softmax as
`Softmax -> IsNaN -> Where(NaN, 0, probability)` to handle fully masked rows.
ORT 1.23.2 MIGraphX does not claim IsNaN, splitting attention into two GPU
partitions and copying the full softmax matrix to CPU and the boolean result
back to GPU. This is more than mask preparation overhead. See
`onnx_validation/pr523/static-rerun/w7900/math/mask-diagnostic.log`.
A mask/export rewrite must preserve the all-masked-row behavior or explicitly
constrain valid inputs before removing that guard.

### Explicitly enable MLIR attention fusion on W7900

The default traces above were collected without this override. To request
MIGraphX's fused attention compiler path on gfx1100, set this **before starting
Python / creating the ORT session**:

```bash
export MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention
unset MIGRAPHX_DISABLE_MLIR
```

This is a MIGraphX compiler environment setting, not a CUDA `sdpa_kernel`
provider option. Use a fresh process and avoid reusing a precompiled model
cache from the default run. The ROCm 7.2 implementation automatically enables
plain MLIR attention for gfx94; gfx1100 needs the explicit request. See
[`mlir_attention_enabled`](https://github.com/ROCm/AMDMIGraphX/blob/rocm-7.2.0/src/targets/gpu/fuse_mlir.cpp#L123)
and [AMD's tuning guide](https://rocmdocs.amd.com/projects/AMDMIGraphX/en/develop/how-to/model-performance-tuning.html).

With the override, FP16 unmasked benchmarks passed at S=128, 512, 1280,
2560; FP32 unmasked models failed compilation and all masked models still
required CPU fallback. At S=2560, FP16 ORT latency fell from 5.825 ms to
2.311 ms (host transfers included). Accuracy MAE was 1.44e-5.

An ORT-only FP16 probe at B=4, S=512, H=16, D=16 confirmed that this override
compiles attention into one MLIR GPU kernel, with no full attention-score
scratch tensor in the final program. Traces are under
`onnx_validation/pr523/static-rerun/w7900/math/force-attention/`.
`MIGRAPHX_FLASH_DECODING_NUM_SPLITS` is a separate split-KV optimization in
this installed release, not the switch needed to enable ordinary attention
fusion. Newer develop documentation uses different flash-decoding settings.

## Save results and release the GPU

From your local repository, before the pod expires:

```bash
bash ngt/session.sh fetch     # copies results to ngt/results/
bash ngt/session.sh delete
```

Both the environment and results under `/tmp` disappear with the pod. For
long runs, use a persistent EOS directory for `--outdir` instead, after
checking EOS authentication; `fetch` assumes the default `/tmp/particleflow/results`.

Sources: [NGT AMD session example](https://gitlab.cern.ch/mlops/ngt/docs/-/blob/master/docs/getting-started.md#amd),
[NGT environment updates](https://gitlab.cern.ch/mlops/ngt/docs/-/blob/master/docs/updates.md),
[AMD ONNX Runtime installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-onnx.html),
[AMD 7.2.1 wheel index](https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/).
