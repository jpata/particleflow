# Portable CLD training benchmarks

The benchmark separates the scientific recipe, the measurement protocol, and the
site runtime. A new accelerator should need a platform YAML and, if necessary,
a small environment adapter rather than a copy of the benchmark driver.

## What is shared and what may differ

| Area | Shared contract | Platform-specific choices |
| --- | --- | --- |
| Scientific recipe | Dataset/config/version, feature engineering, model/decoder, optimizer, seed and loss definitions | None without a separately named comparison |
| Measurement | Batch grid, steps, warmup, calibration, validation exposure, telemetry cadence and metric scopes | Missing hardware counters remain unavailable |
| Numerical mode | Recipe precision and compilation setting | Supported attention implementation, recorded as a runtime difference |
| Input pipeline | Same data inventory and recorded event/hit exposure | Worker count, prefetching, file-sharing strategy and storage location |
| Allocation | One process, one node and one visible device | Slurm account/partition, CPU count, RAM, GPU request syntax |
| Environment | Adapter prepares the stack once and execs Python with arguments intact | Architecture, CUDA/ROCm wheels, container and compiler modules |
| Evidence | Common JSON/CSV fields, units, missing-value semantics, portable plots and checksums | Device name, memory scope, runtime versions and provenance |

Backend and loader differences must stay explicit. Forcing NVIDIA and AMD to use
identical kernels would defeat portability. Conversely, letting a profile change
BF16 to FP32 or the warmup interval would silently change the experiment. The
sweep rejects conflicting precision, compilation and recorder overrides. The
canonical recipe is `configs/training/scenarios/cld_backbone_benchmark.yaml`.
The GH200-named and H100-3.2.1 scenarios remain frozen historical recipes.

## Run on an existing or new platform

Create a profile under `configs/training/platforms/` with one GPU and one task.
Set `data_dir` from an environment variable and define Slurm resources once in
that profile. The scientific recipe fixes BF16 and disables top-level model
compilation; internal compiled kernels may still run. Choose supported attention
and loader settings in `runtime_overrides`. The existing environment adapters are:

- `scripts/empireai/batch_python.sh`: prepared ARM environment.
- `scripts/flatiron/batch_python.sh`: prepared CUDA environment through `uv --no-sync`.
- `scripts/lumi/batch_python.sh`: pinned LAIF container and overlay, preserving Slurm visibility.

An adapter accepts Python arguments and ends with `exec python3 "$@"` (or the
site-equivalent prepared interpreter). It should not install packages per case.
If the allocation already has the right Python environment, omit the adapter.
A new CUDA/ROCm site does not need a Python dispatch-table entry or a new Slurm
worker. An accelerator with a different PyTorch device API still needs compatible
attention, timing and telemetry implementations; the launcher does not provide those.

From the repository root, preview a submission:

```bash
export MLPF_CLD_DATA_DIR=/path/to/reference/tfds
python -m scripts.training.submit_batch_sweep \
  --platform configs/training/platforms/flatiron_h100_single.yaml \
  --python-wrapper scripts/flatiron/batch_python.sh \
  --dry-run \
  --data-manifest configs/training/data/cld_ttbar_hits_config1_3.2.2.json \
  --require-clean-git
```

Remove `--dry-run` to submit the full sweep after qualification. The generic worker starts one `srun` task; the
platform YAML supplies the allocation resources. Replace the profile and adapter
for another site. `--batch-sizes`, `--steps`, `--warmup`, `--output`, `--data-dir`,
`--scenario` and `--case-timeout` pass through to the shared driver. To preview
resolved cases instead of an sbatch command, call the driver directly:

```bash
python -m scripts.training.batch_sweep \
  --platform configs/training/platforms/flatiron_h100_single.yaml \
  --batch-sizes 8 16 --dry-run
```

All site workers now use the same defaults. `MLPF_BATCH_SWEEP_SCENARIO` and
`MLPF_BATCH_SWEEP_OUTPUT` work on every site; explicit CLI arguments win. Without
an output override, the driver uses the profile's experiment directory and a name
containing the platform and Slurm job ID (timestamp outside Slurm). The old site
`run_batch_sweep.sh` files remain compatibility entrypoints with historical
SBATCH directives. Prefer the generic submitter so future resource changes need
only a profile edit.

Real runs freeze the resolved specification along with case/profile YAML. They
record scenario/spec hashes, Git revision and dirty status, dataset preflight
status, and host memory diagnostics on every platform. `--require-clean-git`
rejects dirty or unavailable provenance. Keep outputs in ignored directories or
outside the checkout when using this option. `--data-manifest` checks matching
training dataset selection, filenames/sizes and metadata hashes before any case
starts. It does **not** hash shard payloads. Without the option, the manifest
explicitly says `not_checked`; a successful run is not evidence of dataset identity.
Dry-runs only resolve/preview; they do not verify data or enforce source cleanliness.

For a controlled hardware comparison, use one clean source commit, one copied
dataset, and equivalent recipe settings. Repeat timed cases in varied order and
record complete environment locks/container identities. A new hardware name or
successful import alone does not qualify a platform. Begin with both models at
batch 8, inspect completion and telemetry coverage, then run the full grid.

## Qualification

After the [site environment setup](hpc-batch-sweep.md), qualify both backbones at
batch 8 before using the full grid. Run the submitter in the prepared environment;
for GH200 that requires an ARM allocation. Select the profile and adapter together:

```bash
python -m scripts.training.submit_batch_sweep \
  --platform configs/training/platforms/flatiron_h100_single.yaml \
  --python-wrapper scripts/flatiron/batch_python.sh \
  --data-manifest configs/training/data/cld_ttbar_hits_config1_3.2.2.json \
  --require-clean-git --batch-sizes 8 --steps 20 --warmup 2
```

Inspect `sweep-state.json`: both cases must be `completed`. An overall successful
sweep may contain OOM cases, which are recorded capacity limits. Other errors and
timeouts make the command fail after the remaining cases finish. Inspect validation,
checkpoint writes, nonzero telemetry coverage and telemetry errors. Then omit the
short-case overrides to run the default grid. Use a fresh output directory for each
sweep. Retain Slurm accounting separately for allocation walltime and billed cost.

## Recorder artifacts and metric scope

Recording is controlled by `benchmark.enabled`; the sweep enables it explicitly.
For a scenario run, pass `--set benchmark.enabled=true`. Each worker/session writes
a unique `benchmark/<worker-session>/` directory, including resumed and Ray workers.

| Artifact | Contents |
| --- | --- |
| `metadata.json` | Resolved configuration, package/runtime versions, device/rank identity, source and scheduler provenance |
| `steps.jsonl` | Actual events/hits/sequence lengths, padded work, host step time, loss and sampled CUDA stream timing |
| `telemetry.jsonl` | Device/process-tree measurements, phase/step, collection time and errors |
| `phases.jsonl` | Worker, training, validation, testing, plotting and checkpoint boundaries and checkpoint sizes |
| `summary.json` | Work totals, warmup-excluded throughput, latency quantiles, phase times, memory peaks and covered busy/power intervals |

The recorder defaults to one-second sampling, 100 warmup steps per session and
CUDA timing every 10 steps; sweeps select 20 warmup steps. Zero
`benchmark.gpu_timing_interval` disables CUDA events. Stream elapsed time includes
host submission gaps and is not the sum of kernel times. Host samples cover the
process tree, including loader workers; missing counters remain unavailable.

Regenerate `benchmark-report.json` and `.csv` from a run or scenario with
`python -m mlpf.benchmark_report RUNS_DIRECTORY`. The sweep updates reports in
`runs/` after each case. Actual work sums across ranks; elapsed time uses the
observed worker span. Retries include reprocessed events. Nested phase durations
overlap, so inclusive durations must not be summed. Framework memory peaks can
include validation and differ from sampled device-wide memory.

Energy uses trapezoidal integration between valid, same-phase samples, with
coverage reported explicitly. Device/package power is not process attribution;
check physical boundaries before comparing GPUs. Missing samples are not zero
utilization. GPU busy percentage does not establish SM/Tensor Core activity,
achieved FLOPs, bandwidth or coherent-memory/interconnect performance.

`step_valid_time` now measures the current validation call, while
`step_test_time`, `step_plot_time` and `step_validation_cycle_time` cover the other
phases. Older GH200 histories used an incorrect cumulative `step_valid_time`;
do not use that field from the historical runs to estimate validation overhead.

## Export results without a training environment

The exporter reads raw sweep artifacts and writes compact `results.json`,
`cases.csv`, `steps.csv` and `validation.csv`. It preserves dataset and runtime
configuration, Git provenance, memory/timing scope and failed-case status. It
omits absolute data paths, raw host logs, device UUIDs, checkpoints and prediction
parquet files. Original raw inputs remain in the private/local study archive.
Exporting does not require importing the training model or a GPU.

```bash
python -m scripts.training.export_benchmark \
  --sweep 'GH200=/path/to/gh200_sweep' \
  --sweep 'H100 PCIe=/path/to/h100_sweep' \
  --sweep 'MI250X (1 GCD)=/path/to/lumi_sweep' \
  --release v3.2.0 \
  --output results/benchmark-release/benchmarks/v3.2.0/cld_gpu_batch_sweep
python results/benchmark-release/benchmarks/v3.2.0/cld_gpu_batch_sweep/render_plots.py
```

The copied plotting script requires only pandas/numpy/matplotlib. It draws
throughput, memory, latency-tail and total-unweighted-loss comparisons. It refuses
multiple sessions/repeats per platform/model/batch rather than silently choosing
or averaging them. Validation plots require matching endpoint steps. The initial
plotter covers one device per case; distributed throughput needs explicit work
aggregation and is not inferred here.

Add a release README and any reviewed slides, then regenerate the checksum
manifest with `scripts.training.export_benchmark.write_checksums(package_path)`.
The manifest must cover every file before publication:

```bash
python -m scripts.training.publish_benchmark \
  results/benchmark-release/benchmarks/v3.2.0/cld_gpu_batch_sweep \
  --repo-id jpata/particleflow --tag v3.2.0
# Apply the inspected package:
python -m scripts.training.publish_benchmark \
  results/benchmark-release/benchmarks/v3.2.0/cld_gpu_batch_sweep \
  --repo-id jpata/particleflow --tag v3.2.0 --upload
```

Publishing adds `benchmarks/TAG/PACKAGE_NAME/` to the **model** repository's main
branch, verifies every remote Git/LFS content hash, and creates the tag on that
exact commit. It pins the upload's parent revision and refuses existing tags or
destination files. It never deletes older releases or moves tags. If upload
succeeds but verification/tagging fails, inspect the reported commit before
recovery; do not blindly overwrite the release or repoint an existing tag.

## Historical evidence and remaining limitations

The [v3.2.0 evidence package](https://huggingface.co/jpata/particleflow/tree/5a1fbd8f7cf71c7446a5202f4223009902e2d048/benchmarks/v3.2.0/cld_gpu_batch_sweep)
covers the 29–30 October 2026 runs at an immutable Hugging Face commit. It contains
23 files (4.7 MB), including seven reproducible figures and a 19-slide presentation.
Use the pinned commit link rather than assuming a Git tag exists. The folder
version labels benchmark evidence, not the TFDS dataset version.
H100 used TFDS 3.2.1, GH200/LUMI 3.2.2. The runs used different dirty Git bases,
NVIDIA native versus ROCm external attention, and four versus two loader workers.
MI250X means one GCD, not a whole two-GCD module. These facts cannot be repaired
retroactively by unifying the launcher. The next comparison should use the strict
preflight options and repeat the cases on a single committed revision.

The package retains the original single-run findings and loss curves. It does
not claim convergence, confidence intervals, portable full-system energy scope,
or scheduler-billed cost. LUMI power is unavailable; GH200 Attention batch 32 has
no usable phase-attributed busy/power interval. Missing cgroup readings likewise
must not be interpreted as zero memory use or an absence of OOM events.
