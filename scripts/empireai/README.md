# EmpireAI Grace Hopper training

For the same study on LUMI and Flatiron H100, see the
[cross-platform reproduction runbook](../../docs/training/hpc-batch-sweep.md).

`alphagh01` is an ARM64 GH200 test node in the Alpha Slurm instance. The
`empireai_gh200` profile requests one GPU using account `cuny`, QoS `burst`,
partition `grace`, constraint `gracehopper`, and reservation `ghtest`.
The reservation currently expires on 2026-11-27; update the profile when it changes.

The 2026-10-29 probe found 72 CPU cores, 64 KiB pages, a GH200 with 97,871 MiB
GPU memory, driver 595.45.04, and host Python 3.12. `/mnt/home` and `/projects`
are mounted; `/mnt/lustre` and `/ddn` are absent on this particular node.
The [storage guide](https://empireai.freshdesk.com/en/support/solutions/articles/157000376738-empire-ai-storage-architecture-user-workflow-guide)
recommends reading training data directly from `/projects`. It does not document
a way to mount the old Alpha `/mnt/lustre` tree on Grace.

## Install the ARM environment

From the repository on the login node:

```bash
srun --partition=grace --account=cuny --qos=burst --reservation=ghtest \
  --constraint=gracehopper --cpus-per-task=8 --mem=32G --time=00:30:00 \
  bash scripts/empireai/setup_env.sh
```

Setup installs uv, Python 3.11, and the isolated
`envs/empireai-gh200/.venv` under the repository. Binaries and caches are ARM-specific.
The checked-in environment lock pins the resolved packages; subsequent setup uses
`--locked`. The environment imports the repository through `PYTHONPATH` instead of
installing the root project's x86-only dependency sources. It supports the
attention and HEPTv2 training paths with PyTorch SDPA and native packed varlen
attention for the set decoder (zero attention dropout on NVIDIA). External `flash-attn`,
`torch-scatter`, LitePT/spconv, data-production tools, and Ray HPO are not installed.
Do not run root `uv run` or reuse this virtual environment on the x86 login node.

## Copy a smoke-test dataset

The smoke scenario uses only ttbar hits config `1`, version `3.2.2` (about 30 GiB).
The directory contains its complete TFDS train and test splits, including metadata.
Copy it from the login node, which has access to the source Lustre filesystem:

```bash
mkdir -p data/empireai-smoke/tfds/cld_edm_ttbar_hits
rsync -a --stats \
  /mnt/lustre/cuny/jpata/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds/cld_edm_ttbar_hits/1 \
  data/empireai-smoke/tfds/cld_edm_ttbar_hits/
```

This home-directory copy is for qualification. Account for it and the environment
within the home quota. The full dataset should use compute-visible project storage.
Set `MLPF_CLD_DATA_DIR` to override the default repository-local TFDS root; export it
before both preview and submission. The worker checks data-directory access before
starting the pipeline. Paths in the profile expand using environment variables, so
source `scripts/empireai/environment.sh` for direct Python invocations as well.

## Preview and run

The ARM environment must also resolve submissions. Obtain a short CPU allocation:

```bash
srun --partition=grace --account=cuny --qos=burst --reservation=ghtest \
  --constraint=gracehopper --cpus-per-task=2 --mem=8G --time=00:10:00 \
  bash scripts/empireai/train_scenario.sh cld_gh200_smoke gh200 --dry-run
```

Remove `--dry-run` to submit the smoke job. It runs 20 steps at global batch 2,
validates and checkpoints every 10 steps, and uses four validation/test events.
Logs are in `logs_slurm/`; checkpoints and the resolved scenario manifest are in
`experiments/cld_gh200_smoke/`. The worker inherits the repository working directory
from Slurm and does not activate x86 modules or synchronize dependencies.

Use `--continue` to resume an interrupted compatible run. A completed run is skipped.
For a direct worker invocation inside a GPU allocation:

```bash
bash scripts/empireai/run_scenario.sh \
  configs/training/scenarios/cld_gh200_smoke.yaml \
  configs/training/platforms/empireai_gh200.yaml
```

After qualification, use a full-data path and an existing scientific scenario, such
as `cld_pf_heptv2_default_50k`. That scenario requires all its configured samples,
not just this copied subset, and its batch 64 must be qualified for memory first.

## Attention versus HEPTv2 benchmark

`cld_gh200_backbone_benchmark` is the initial comparison on the copied ttbar
hits subset. It derives both CLD hit variants from `clic_cld_backbone_benchmark`:
six backbone layers, width 256, 16 backbone heads, the same four-layer/256-slot
set decoder, tracker momentum readout, input features, data, optimizer and seed.
The runner rejects differences outside the backbone configuration.

Both variants run 1,000 steps at global batch 8, seed 12345, BF16, LAMB,
LR 0.001 and cosine decay. Validation/checkpoints run every 250 steps with 128
validation events and 32 test events. Training uses all 90,000 copied training
events; each run presents 8,000 events. This is an initial single-seed comparison,
not a convergence study. HEPTv2 activation checkpointing follows the original
benchmark recipe; its measured runtime includes recomputation. Parameter counts
are not matched. Attention uses PyTorch jagged Flash SDPA and both decoders use
native PyTorch varlen attention.

The batch-8, 20-step qualification of both variants completed successfully under
`experiments/gh200_qualification/cld_gh200_backbone_benchmark/`; its combined log
is `logs_slurm/gh200_backbone_qualification.log`.

Submit from an ARM allocation using the picker:

```bash
bash scripts/empireai/train_scenario.sh cld_gh200_backbone_benchmark gh200
```

The two array tasks share the single reserved GPU sequentially. To report progress
or final results from the ARM environment:

```bash
python3 scripts/empireai/summarize_benchmark.py \
  experiments/cld_gh200_backbone_benchmark
```

This writes `benchmark-summary.json` inside the scenario directory. It reports
training-step throughput from logged samples after step 100 (excluding task-weight
calibration), observed peak PyTorch GPU allocation, and unweighted validation loss.
Step timing includes data loading, forward/backward and optimizer work, and excludes
validation/checkpoint work. GPU memory maxima can include validation. Total job
walltime should be reported separately when comparing cost. Weighted total losses
are not directly comparable because each model calibrates its own task weights.

The initial campaign is Slurm array `153194`: task 0 is attention and task 1 is
HEPTv2. A dependent CPU report job runs `scripts/empireai/report_benchmark.sh`
after both tasks finish (including partial results if a task fails).

## Utilization evidence for an allocation request

### Batch-size sweeps

Submit `sbatch scripts/empireai/run_batch_sweep.sh` from the repository to run
attention and HEPTv2 at batches 8, 16, 32, 64 and 128 on `ghtest`. The generic
driver is `scripts/training/batch_sweep.py`; the worker forwards CLI options such
as `--batch-sizes 8 16 32 --steps 120 --warmup 20`.

Each case runs in a fresh process, with 120 steps, 20 warmup/calibration steps,
and validation/checkpointing at the end. Training settings and input features
follow `cld_gh200_backbone_benchmark`. This is a throughput/capacity comparison;
larger batches present more training events, so final losses are not matched
scientific-quality comparisons. Out-of-memory cases are recorded as capacity
limits and do not stop the sweep. Unexpected errors/timeouts are recorded and
make the overall job fail after the remaining cases run. Timeouts terminate the
case's entire process group. The default per-case timeout is one hour and the
GH200 worker allocation has a four-hour limit.

Artifacts live under `experiments/gh200_batch_sweep_<jobid>/`: frozen YAML cases,
per-case logs, atomic `sweep-state.json`, and experiment data under `runs/`.
The driver regenerates `runs/benchmark-report.json` and `.csv` after each case;
these include explicit variant and global-batch columns. Job `153260` is the
initial sweep using these defaults. A dry-run resolves every case without writing
or launching training; four sweep-specific tests cover configuration invariants,
failure classification, process timeouts and the dry-run.

### Portable benchmark recorder

New training runs collect portable metrics by default through
`mlpf/model/benchmarking.py`. Each worker/session creates a unique directory
under its experiment's `benchmark/`, so ranks and resumed jobs do not overwrite
one another. Disable with `--set benchmark.enabled=false` in the scenario runner.

| Artifact | Contents |
| --- | --- |
| `metadata.json` | Resolved configuration, versions, GPU identity, rank/world size, Slurm identity, git revision/diff hash and untracked-source hashes |
| `steps.jsonl` | Every completed training batch's actual events/hits/sequence lengths, tensor bytes, host-wall latency, losses, sampled diagnostics and asynchronous CUDA stream timings |
| `telemetry.jsonl` | Timestamped device and process-tree readings, phase and step, collection duration, and explicit collection errors |
| `phases.jsonl` | Worker/training-loop, validation, test, plotting and checkpoint boundaries; saved checkpoint byte counts |
| `summary.json` | Work totals, warmup-excluded throughput, latency quantiles, inclusive/exclusive phase times, GPU busy/energy estimates with coverage, framework/device memory and final status |

Defaults are one-second telemetry, 100 recorded warmup steps per session, and
asynchronous CUDA timing every 10 steps. For short qualification runs use
`--set benchmark.warmup_steps=2`. `benchmark.sample_interval_seconds` and
`benchmark.gpu_timing_interval` tune sampling; zero disables CUDA event timing.
The recorder does not add per-step device synchronization; the existing detailed
TensorBoard diagnostics still synchronize logged steps. Redundant synchronous SMI
snapshots are suppressed while the continuous recorder is enabled.

NVIDIA SMI records GPU/memory busy percentages, device memory, power/limit,
temperature, clocks, performance state and throttle-reason mask. ROCm SMI records
available busy/memory/power/temperature/clock fields. Unsupported values are
missing rather than fabricated zero readings. CPU-only nodes and missing SMI tools
still record host and work metrics. Host samples include process-tree RSS/PSS,
CPU core equivalents relative to affinity capacity, cumulative CPU time and I/O
counters; exited workers with unobserved CPU tails are flagged. Pinned allocator
statistics are included in sampled framework diagnostics when PyTorch exposes them.

Summarize a run or an entire scenario from any compatible Python environment:

```bash
python3 -m mlpf.benchmark_report experiments/gh200_metrics_verified
```

This creates portable `benchmark-report.json` and `.csv`. Actual events are summed
across ranks; elapsed time uses the observed worker span. Recorded GPU-worker hours
exclude launcher/import time before recorder startup and are not Slurm billing.
Retries count reprocessed events as compute work. Energy is trapezoidally estimated
only between consecutive valid, same-phase readings; gaps and uncovered boundaries
are reported, not extrapolated. Power is vendor-reported device/package scope,
not process-attributed energy or necessarily the same physical boundary across GPUs.
CUDA stream intervals include host submission gaps and are not summed kernel time.
SM/Tensor Core activity, bandwidth, coherent-memory and interconnect counters require
separate DCGM/Nsight/rocprof collection and are explicitly unavailable in metadata.

Validation timing is fixed for new runs: `step_valid_time` measures the current
validation call, with separate `step_test_time`, `step_plot_time`, and
`step_validation_cycle_time`. `elapsed_training_loop_seconds` identifies cumulative
elapsed time explicitly. The older measurements discussed below predate this fix.

Verification: unit/regression tests cover both SMI adapters, missing metrics,
UUID/mask selection, energy/coverage aggregation, shutdown/error paths, real work,
resume warmup, distributed totals, and validation timing. Live GH200 attention and
HEPTv2 checks each completed 20 steps/160 events with 10 CUDA timing records,
40/47 valid GPU samples and zero telemetry errors. Evidence is under
`experiments/gh200_metrics_verified/` and `logs_slurm/gh200_metrics_verified.log`.
ROCm parsing is fixture-tested; no AMD hardware was available for live validation.

### Historical batch-8 baseline

The completed 1,000-step jobs (`153194_0`, `153194_1`) saved utilization in
TensorBoard. Extract the evidence from an ARM environment with:

```bash
python3 scripts/empireai/analyze_utilization.py \
  experiments/cld_gh200_backbone_benchmark
```

Outputs are `gpu-utilization.json`, `gpu-utilization-scalars.csv`, and
`gpu-utilization.png` in that scenario directory. After warmup and excluding
windows that span validation, the mean of 87 utilization-window means is 48.84%
for attention and 52.69% for HEPTv2. The largest individual sampled GPU busy values
are 100% and 99%. The monitor samples `nvidia-smi` about every 0.5 seconds and
resets its accumulated statistics at each logging step. It does not save sample
counts per window, so these means are not exact time-weighted job averages.
GPU busy time is not SM occupancy, Tensor Core activity, or achieved FLOPs.

Mean logged loader wait is 1.18/1.20 ms, versus 846.68/919.66 ms per logged
training step. The logged batches have identical mean hit counts (76,797 hits),
giving approximately 90,704/83,507 hits per second for attention/HEPTv2.
These are sampled, instrumented steps; full-job useful throughput is separately
8,000 event presentations divided by 727/785 seconds, or 11.00/10.19 events/s.
Logging/profiling adds work to the sampled steps, so their throughput should not
be presented as the exact mean of all training steps.

Slurm's 30-second accounting reports GPU-utilization high-water fields of 51/52
and process GPU-memory high-water fields of 14,128/15,696 MiB. The PyTorch tensor
allocation peaks (6,185/7,147 MiB) have a different scope. Neither Slurm's
`TRESUsageInAve` nor a transient 100% GPU busy sample should be quoted as sustained
compute efficiency. See the [Slurm accounting semantics](https://slurm.schedmd.com/gres.html#Accounting)
and [NVIDIA utilization definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization).

Before a larger allocation request, prioritize:

1. A real-data batch sweep (8, 16, 32, 64, then larger if memory permits), logging
   continuous timestamped utilization, power, memory, and phase boundaries. Record
   events/s, hits/s, sequence-length distributions, p50/p95 step times and both
   framework and total device memory. Use repeated measurements after warmup.
2. Short Nsight Systems traces to locate CPU/kernel-launch gaps and synchronization;
   use DCGM or separate kernel profiles for SM/Tensor Core activity and memory
   bandwidth. Reduce instrumentation overhead for production throughput runs.
3. An identical H100/H200 baseline to establish a Grace Hopper benefit: useful
   throughput, time/GPU-hours to a scientific quality target, or the ability to
   train larger realistic events/models. Current results establish ARM portability,
   not a measured Grace CPU/coherent-memory advantage or a large-HBM requirement.
4. Representative long training and several seeds, with particle efficiency/F1,
   charged-momentum and jet-resolution metrics alongside unweighted loss. Compare
   at matched event exposure when changing batch size.
5. If requesting distributed jobs, measure strong/weak scaling and communication
   overhead on actual multi-GPU hardware. This one-GPU reservation cannot establish
   scaling. Independent seeds/variants can instead justify concurrent single-GPU jobs.
6. A campaign budget from measured full-run GPU-hours, model/seed counts, evaluation
   and checkpoint overhead, storage, and retries. Test production storage access;
   the copied home subset is not a production filesystem benchmark. Collect power
   continuously if reporting joules/event; the saved data cannot reconstruct energy.

The historical `history/step_*.json` field `step_valid_time` subtracted only
the latest training-step duration from elapsed time since training began. It is
not the duration of that validation cycle; do not use those old values to estimate
validation overhead. New runs use the corrected phase timers described above.

## Smoke-test verification

The native environment imports `mlpf.pipeline` and passes a BF16 CUDA
forward/backward check on the GH200. The scenario and submission suites pass
(75 tests), as do 52 existing set-prediction tests and two NVIDIA output/gradient
checks against FP32 math attention. The latter cover masked/empty events and both
zero-dropout training and evaluation with dropout disabled.

On 2026-10-29, job `153187` completed the 20-step smoke recipe, validation at
steps 10 and 20, predictions, and checkpoint writes using native PyTorch varlen
attention with no external `flash-attn` installed. A second run restored
`checkpoint-10.pth`, resumed at step 11, and reached step 20. Evidence is in
`logs_slurm/log_cld_gh200_smoke_153187_0.err` and
`logs_slurm/gh200_smoke_resume.log`; the initial full run's artifacts are under
`experiments/cld_gh200_smoke/heptv2_seed12345_20261029_040554_965395/`.

When running the scenario tests on EmpireAI, unset the exported `module`
function and set `BASH_ENV=/dev/null` so Lmod does not override shell-test stubs:

```bash
source scripts/empireai/environment.sh
unset -f module
export BASH_ENV=/dev/null
python3 -m pytest -q tests/test_training_submission.py tests/test_training_scenarios.py
```
