# Reproducing the attention export study

Run commands from the repository root unless otherwise noted. All environments are isolated under `/tmp`; the application training environment is not changed.

## Provisioning

NGT authentication and namespace setup follow `ngt/README.md`. The study uses `ngt/session-l40s.yaml` and the existing `sdpa-w7900` pod. `profiling: "true"` is set on the full L40S session. Both manifests use a finite lifetime. Copy results before the sessions expire.

```bash
bash ngt/kubectl.sh create -f ngt/session-l40s.yaml
bash ngt/kubectl.sh get pods -o wide
```

`ngt/kubectl.sh` uses the repository's explicit kubeconfig and username namespace. It does not change the default kubeconfig. Check GPU capacity and wait for Running before executing commands.

Inside the L40S pod:

```bash
sudo dnf install -y python3.12 python3.12-pip
python3.12 -m venv /tmp/attention-venv
/tmp/attention-venv/bin/pip install torch==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
/tmp/attention-venv/bin/pip install numpy==1.26.4 onnx==1.19.1 \
  onnxscript==0.5.3 onnx_ir==0.1.12 psutil matplotlib graphviz
/tmp/attention-venv/bin/pip install onnxruntime-gpu==1.29.0 \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
/tmp/attention-venv/bin/pip install --no-deps --target /tmp/attention-ort123 \
  onnxruntime-gpu==1.23.2
```

Use the AMD environment described in `ngt/README.md`, including native ROCm/MIGraphX libraries and the compatible C++ runtime. Pin the same ONNX/ONNX Script/ONNX IR versions. Do not install a different host GPU driver. The AMD ORT wheel requires newer C++ ABI symbols than the base Alma9 library. The existing pod uses an isolated Conda runtime prefix containing libstdcxx/libgcc 14.4.0. With Conda or micromamba available, the core observed builds can be installed separately:

```bash
micromamba create -y -p /tmp/sdpa-cxx-runtime -c conda-forge \
  libstdcxx=14.4.0=h934c35e_7 libgcc=14.4.0=hecc6394_7
export LD_LIBRARY_PATH="/tmp/sdpa-cxx-runtime/lib:/opt/rocm-7.2.4/lib:/opt/rocm-7.2.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The [complete auxiliary runtime inventory and package hashes](data/cxx-runtime.json) preserve the exact recorded prefix packages, including OpenMP and compatibility metapackages. The raw provenance records describe the tested environment more precisely than a generic setup recipe.

## Canonical models

Copy `scripts/attention-study/` to `/tmp/particleflow/scripts/` in each pod. The exact shared model bundle from this run is in `onnx_validation/attention-study/bundle/`.

To generate it again, in the configured environment:

```bash
cd /tmp/particleflow
python scripts/attention-study/study.py export \
  --out results/attention-study/bundle
python scripts/attention-study/finalize_bundle.py \
  results/attention-study/bundle
```

`finalize_bundle.py` materializes the dense Attention-24 mask query axis and adds adapted modern padded graphs while preserving raw modern exports. Run it once on a freshly exported bundle. `modern_export.py` is a diagnostic retry helper used during setup, not required with the pinned compatible dependencies.

Copy the resulting bundle to the other pod without re-exporting it there. Each manifest row includes a SHA-256 hash, export status, shape, precision, and node counts. Source scripts are also hashed in the provenance. The first run's failed modern-export conversion attempts are retained separately and excluded from performance results.

## Timing

For each backend, run the same command with the appropriate Python executable and provider:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/attention-study/study.py run \
  --bundle results/attention-study/bundle \
  --out results/attention-study/cuda129 \
  --provider cuda --warmup 5 --iters 10
```

There are three timing trials per case; `--iters 10` means thirty measured calls in total. Use `cpu` for CPUExecutionProvider and `migraphx` for the AMD pod. Use separate output directories for each configuration.

For the AMD default run, leave `MIGRAPHX_MLIR_USE_SPECIFIC_OPS` unset. For the fused-attention request, start a new process with:

```bash
export MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention
unset MIGRAPHX_DISABLE_MLIR
```

For the version-matched CPU/CUDA control, use the CUDA venv with `PYTHONPATH=/tmp/attention-ort123` and add `--quick`. It tests sequence 512 static models and records the loaded ORT version, rather than trusting the package inventory alone.

The GPU benchmark disables CPU fallback. Dynamic-shape failures can therefore include host shape operations; the separate probes below allow CPU fallback and record actual placement. Rows that fail during setup, execution, correctness, or bound-I/O validation remain in `results.json`.

## Profiling, dynamic shapes, and memory

Run these after timing has completed, with no concurrent study workload on the device:

```bash
export STUDY_PYTHON=/tmp/attention-venv/bin/python
bash scripts/attention-study/diagnostics.sh cuda
# In the AMD pod, with its configured library paths:
export STUDY_PYTHON=/tmp/sdpa-w7900-venv/bin/python
bash scripts/attention-study/diagnostics.sh migraphx
```

The diagnostics collect Nsight Systems kernel summaries on CUDA, rocprofv3 traces and final MIGraphX programs on AMD, dynamic-size call sequences, and separate memory probes. The recorded AMD disk-cache scenario produced a GPU memory-access fault; the reproducer skips that scenario by default. Set `STUDY_TEST_DISK_CACHE=1` only for a deliberate repeat of that diagnostic. Its original failure log remains in the evidence bundle. The ORT-only probes execute no PyTorch reference GPU computation. Compiler tuning kernels can appear in trace summaries; inspect the final inference dispatches and the final compiled program rather than treating tuning work as inference.

`memory.py` samples CPU process RSS or global GPU driver usage every 2 ms in a separate process. Report deltas above the pre-session baseline, include context/provider caches, and treat sampled peaks as lower bounds. Do not combine CPU RSS and GPU VRAM into one directly comparable hardware metric.

## Balanced batch sweep

```bash
python scripts/attention-study/batch_bundle.py \
  results/attention-study/batch-bundle
python scripts/attention-study/study.py run \
  --bundle results/attention-study/batch-bundle \
  --out results/attention-study/cuda-batch \
  --provider cuda --warmup 5 --iters 10
```

Distribute this bundle unchanged to AMD. It varies batches 1, 2, 4, 8, 16 at sequence lengths 512 and 1280, with FP16 explicit safe decomposition and Attention-24 valid lengths. Exact per-example lengths and valid-token counts are recorded.

## Scoped application override

The study includes the PR #523 wrapper snapshot as `scripts/attention-study/mlpf-onnx-attention.py`. Run:

```bash
python scripts/attention-study/override_demo.py \
  --wrapper scripts/attention-study/mlpf-onnx-attention.py \
  --out results/attention-study/override-demo
```

This imports the actual repository wrapper, exercises its placeholder-to-standard-op lowering, checks output against PyTorch MATH, and verifies the wrapper rejects empty examples and masks with holes. This is a module/export contract test, not a trained full-model benchmark.

## Rendering

Fetch `results/attention-study/` from both pods into `onnx_validation/attention-study/`. Keep configuration subdirectories distinct. The L40S and W7900 execute identical bundle files; compare their manifest hashes before aggregating results.

A separate plotting environment needs NumPy, Matplotlib, ONNX, the Python graphviz package, and the `dot` executable:

```bash
python scripts/attention-study/audit.py onnx_validation/attention-study
python scripts/attention-study/report.py \
  --root onnx_validation/attention-study \
  --docs docs/attention-study
```

SVG figures and text dumps are suitable for inspection and sharing; PDF and PNG versions are generated for the statistical plots. Raw Nsight and rocprof data remain in the local evidence directory. None of these commands publishes the article or sends results to an external service.

Additional figure/evidence builders:

```bash
python scripts/attention-study/evidence_figures.py --root onnx_validation/attention-study --docs docs/attention-study
python scripts/attention-study/runtime_dump.py --root onnx_validation/attention-study --docs docs/attention-study
python scripts/attention-study/finalize_article.py --root onnx_validation/attention-study --docs docs/attention-study
```

Run `python scripts/attention-study/test_contracts.py` in either configured vendor environment for the numerical rewrite/export tests. `audit.py` checks canonical model hashes against both files and recorded experiment rows, validates ONNX files, checks unique cases and strict GPU node placement, and writes separate validated classifications without editing raw measurements.
