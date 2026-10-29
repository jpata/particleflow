# Repeating the CLD hit-model batch sweep on HPC systems

The same study runs on EmpireAI GH200, Flatiron H100 and LUMI MI250X. The shared
driver is [`scripts/training/batch_sweep.py`](../../scripts/training/batch_sweep.py).
Its scenario retains the historical name `cld_gh200_backbone_benchmark`, but the
science configuration is hardware-independent. Dedicated **single-device** profiles
and site workers select the environment and scheduler resources.

## Study definition

| Setting | Value |
| --- | --- |
| Data | `cld_edm_ttbar_hits/1/3.2.2`: 90,000 train, 10,000 test events |
| Models | Attention and HEPTv2; six backbone layers, width 256, 16 heads |
| Shared decoder | Four layers, 256 slots, global Flash attention, tracker momentum readout |
| Features | Geometry, calorimeter/tracker neighborhoods; identical across models |
| Optimizer | LAMB, LR 0.001, cosine schedule, BF16, seed 12345 |
| Per-device batches | 8, 16, 32, 64, 128 |
| Per case | 120 steps; first 20 steps excluded from steady throughput |
| Loss calibration | First 20 steps |
| Validation/checkpoint | End of each case; 32 validation and 8 test events |
| Instrumentation | Continuous telemetry; CUDA/HIP event timing every 10 steps; detailed logging every 20 |
| Isolation | Fresh Python process per case; OOM recorded and sweep continues |

HEPTv2 uses activation checkpointing, matching the reference recipe; attention
does not. Cases process different event counts as batch size changes. This is a
throughput/memory-capacity study, not a matched-exposure comparison of final losses.
Use the same branch commit, data copy and settings on all sites. Do not substitute
the existing eight-device `lumi_mi250x` or `flatiron_h100` profiles.

## Clone the branch

```bash
git clone --single-branch --branch jp_20261029_gracehopper \
  https://github.com/jpata/particleflow.git
cd particleflow
git rev-parse HEAD
mkdir -p logs_slurm
```

For an existing checkout, switch to the same branch and update it before running.
Code and the small reference manifests are versioned; datasets, environments,
checkpoints and raw experiment results are not included in the clone. The ARM64
virtual environment from EmpireAI must not be copied to x86 LUMI/Flatiron.

## Use the exact data subset

`MLPF_CLD_DATA_DIR` is the TFDS **root**, with this tree underneath it:

```text
$MLPF_CLD_DATA_DIR/
  cld_edm_ttbar_hits/1/3.2.2/
    dataset_info.json
    features.json
    metadata.json
    cld_edm_ttbar_hits-train.array_record-00000-of-01024
    ...
    cld_edm_ttbar_hits-test.array_record-00000-of-00128
    ...
```

The reference subset contains 1,155 files and 31,326,294,015 bytes (about 30 GiB).
If the site's existing TFDS root already contains this copy, use it directly.
Otherwise transfer config `1` from EmpireAI using your usual authenticated transfer
method. For example, run this on the **EmpireAI login node**, where Lustre is mounted:

```bash
export TARGET_LOGIN=your-user@your-target-login
export TARGET_TFDS=/scratch/your-project/your-user/mlpf-benchmark/tfds
ssh "$TARGET_LOGIN" "mkdir -p '$TARGET_TFDS/cld_edm_ttbar_hits'"
rsync -a --stats \
  /mnt/lustre/cuny/jpata/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds/cld_edm_ttbar_hits/1 \
  "$TARGET_LOGIN:$TARGET_TFDS/cld_edm_ttbar_hits/"
```

On the target, set `MLPF_CLD_DATA_DIR` and check it:

```bash
export MLPF_CLD_DATA_DIR=/scratch/your-project/your-user/mlpf-benchmark/tfds
python3 scripts/training/check_batch_data.py "$MLPF_CLD_DATA_DIR"
```

This checks every filename/size against the versioned inventory and hashes the
three metadata files. It does not read/checksum all shard payloads. A separately
regenerated TFDS dataset can differ even with the same version label; transfer
the reference copy for a direct comparison. Keep the complete train/test shards,
not just a handful of files with unchanged metadata.

## LUMI: one MI250X GCD

LUMI exposes each MI250X graphics compute die as one Slurm/HIP GPU with 64 GB HBM;
one physical module contains two GCDs. Request one GCD on `small-g`, not an entire
eight-GCD node on `standard-g`. See the official [hardware description](https://docs.lumi-supercomputer.eu/hardware/lumig/)
and [partition guidance](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/partitions/).

The new files are:

- [`lumi_mi250x_single.yaml`](../../configs/training/platforms/lumi_mi250x_single.yaml): one GCD, seven CPU cores, 60G host RAM.
- [`batch_python.sh`](../../scripts/lumi/batch_python.sh): LAIF container plus project overlay.
- [`run_batch_sweep.sh`](../../scripts/lumi/run_batch_sweep.sh): four-hour Slurm sweep worker.

The default account/project is `project_465001293`, following the existing LUMI
workflow. If using another project, update the profile and pass the matching
`sbatch --account=...`; set `LUMI_PROJECT_ROOT` for its bind mount and choose an
explicit output directory. Export the variables below before setup and submission:

```bash
export LUMI_PROJECT_ROOT=/scratch/project_465001293
export MLPF_CLD_DATA_DIR="$LUMI_PROJECT_ROOT/$USER/mlpf-benchmark/tfds"

# Select an immutable, fully resolved image path before creating the overlay.
export IMG=$(readlink -f /appl/local/laifs/containers/lumi-multitorch-latest.sif)
export LUMI_VENV="$PWD/particleflow-laif-env"
bash scripts/lumi/setup_env.sh
bash scripts/lumi/batch_python.sh -m pip freeze > logs_slurm/lumi-pip-freeze.txt
```

The setup installs project requirements into a `--system-site-packages` overlay.
It preserves LAIF's PyTorch, ROCm and FlashAttention stack. Do not run the root
CUDA `uv sync` on LUMI, and do not install the root project's NVIDIA wheel URLs
into this overlay. Pin the same `IMG` for setup and execution; recreate an overlay
if changing the base image. The official [LAIF instructions](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/)
explain the container bindings and available software releases.

Check imports and the ROCm attention backend (these checks do not need a GPU):

```bash
bash scripts/lumi/batch_python.sh -c '
import mlpf.pipeline
import torch
from mlpf.model.set_decoder import flash_attn_varlen_func
print("PyTorch", torch.__version__, "HIP", torch.version.hip)
assert torch.version.hip is not None
assert flash_attn_varlen_func is not None, "Select a LAIF image with working ROCm FlashAttention"
'
bash scripts/lumi/batch_python.sh scripts/training/check_batch_data.py "$MLPF_CLD_DATA_DIR"

bash scripts/lumi/batch_python.sh scripts/training/batch_sweep.py \
  --scenario configs/training/scenarios/cld_gh200_backbone_benchmark.yaml \
  --platform configs/training/platforms/lumi_mi250x_single.yaml \
  --output "$LUMI_PROJECT_ROOT/$USER/particleflow/experiments/preview" --dry-run
```

The dry-run must resolve ten cases with one device and per-device batches 8–128.
It writes no output. Then qualify both models at batch 8:

```bash
export MLPF_BATCH_SWEEP_OUTPUT="$LUMI_PROJECT_ROOT/$USER/particleflow/experiments/lumi_qualification_$(date +%Y%m%d_%H%M%S)"
sbatch --time=00:20:00 scripts/lumi/run_batch_sweep.sh \
  --batch-sizes 8 --steps 20 --warmup 2
```

Confirm **both cases** have status `completed` in `sweep-state.json`, not merely a
successful overall Slurm exit (the sweep intentionally tolerates OOM cases).
Check that telemetry has a nonzero sample count and inspect any telemetry errors.
The worker preserves Slurm's `ROCR_VISIBLE_DEVICES`; it does not apply the existing
full-node eight-device visibility workaround.

For the full sweep, use a fresh output directory or the worker's job-ID default:

```bash
unset MLPF_BATCH_SWEEP_OUTPUT
sbatch scripts/lumi/run_batch_sweep.sh
```

The default output is
`/scratch/project_465001293/$USER/particleflow/experiments/lumi_batch_sweep_<jobid>/`.
On ROCm this study uses the container's external FlashAttention; the GH200 study
used PyTorch 2.11 native NVIDIA attention. Treat software-stack differences as
part of the measured platform result and retain the recorded versions.

## Flatiron: one H100

Use [`flatiron_h100_single.yaml`](../../configs/training/platforms/flatiron_h100_single.yaml)
and [`scripts/flatiron/run_batch_sweep.sh`](../../scripts/flatiron/run_batch_sweep.sh).
They request one H100, eight CPU cores and 128G host RAM, using the repository's
existing Flatiron module stack and CUDA 12.8 environment.

```bash
export MLPF_CLD_DATA_DIR=/mnt/ceph/users/$USER/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds

# Create/sync once; worker and per-case Python processes do not sync packages.
module --force purge
module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer
uv sync
bash scripts/flatiron/batch_python.sh -m pip freeze > logs_slurm/h100-pip-freeze.txt

bash scripts/flatiron/batch_python.sh -c 'import mlpf.pipeline; import torch; print(torch.__version__, torch.version.cuda)'
bash scripts/flatiron/batch_python.sh scripts/training/check_batch_data.py "$MLPF_CLD_DATA_DIR"
bash scripts/flatiron/batch_python.sh scripts/training/batch_sweep.py \
  --scenario configs/training/scenarios/cld_gh200_backbone_benchmark.yaml \
  --platform configs/training/platforms/flatiron_h100_single.yaml \
  --output experiments/h100_preview --dry-run

export MLPF_BATCH_SWEEP_OUTPUT="$PWD/experiments/h100_qualification_$(date +%Y%m%d_%H%M%S)"
sbatch --time=00:20:00 scripts/flatiron/run_batch_sweep.sh \
  --batch-sizes 8 --steps 20 --warmup 2

# After successful qualification:
unset MLPF_BATCH_SWEEP_OUTPUT
sbatch scripts/flatiron/run_batch_sweep.sh
```

The Flatiron helper uses `uv run --no-sync` and the frozen per-case profile uses
`python3`, so subprocesses inherit that environment without repeatedly resolving
dependencies. If your site requires an account, pass `sbatch --account=...` and
update the profile's Slurm account for matching provenance. Check that the selected
partition/constraint still identify H100 nodes before submission.

## Reports, monitoring and comparison

Each job runs ten cases serially on one device. Defaults are four hours per job
and one hour per case. Override the job walltime with `sbatch --time=...` if
needed. The output directory must be fresh; changing `MLPF_BATCH_SWEEP_OUTPUT`
chooses another destination. `--batch-sizes`, `--steps`, `--warmup` and
`--case-timeout` are forwarded to the common driver.

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
export SWEEP_DIR=/path/to/the/printed/job-ID/output/directory
cat "$SWEEP_DIR/sweep-state.json"
```

Results update after each case:

- `sweep-state.json`: completed/running/OOM/error cases, command lines and wall timestamps.
- `cases/`: frozen scientific scenario and resolved platform YAML.
- `logs/`: one training log per variant/batch/seed.
- `runs/benchmark-report.json` and `.csv`: explicit variant/batch, throughput, memory and telemetry summaries.
- `runs/<scenario>/<run>/benchmark/<worker-session>/`: raw timestamped records, metadata and detailed summaries.

Regenerate a report with the appropriate helper:

```bash
bash scripts/lumi/batch_python.sh -m mlpf.benchmark_report /path/to/sweep/runs
# Flatiron:
bash scripts/flatiron/batch_python.sh -m mlpf.benchmark_report /path/to/sweep/runs
```

If ROCm SMI is unavailable inside the container, host/work metrics still work but
GPU telemetry is unavailable. Inspect `metadata.json` and `telemetry.jsonl`; missing
measurements are not zero utilization. Slurm/ROCm busy metrics, SM activity and
Tensor Core activity have different meanings. Device/package power boundaries
and MI250X GCD/module boundaries also differ; do not compare energy without
checking the metric scope. Use events/s **and hits/s**, actual sequence lengths,
memory, covered telemetry intervals, and full Slurm walltime/GPU-hours.

### GH200 reference

The versioned [reference JSON](../../configs/training/references/gh200_batch_sweep_153260.json)
contains the completed EmpireAI sweep (`153260`, 42m 4s). Its steady training rates:

| Batch | Attention events/s | HEPTv2 events/s |
| --- | ---: | ---: |
| 8 | 12.57 | 11.60 |
| 16 | 14.16 | 13.47 |
| 32 | 14.55 | 13.83 |
| 64 | 13.84 | 13.08 |
| 128 | OOM | OOM |

Attention batch-32 utilization is unavailable: its two raw readings crossed phase
boundaries and were excluded from phase-attributed summaries. Its throughput and
framework memory peak are valid. The reference is an end-to-end platform result,
not a pure peak-FLOP measurement. The full raw GH200 experiment directory remains
on EmpireAI and is not distributed in Git.

## Validation status

The shared driver and recorder were run on the GH200, including timeout/OOM
handling, both model backbones, metrics and checkpoint resume. Cross-platform
profiles are checked against the same scientific invariants. Site shell tests use
stubs to verify resource requests, argument forwarding and visibility preservation;
they do not establish that remote modules, accounts or container versions are
currently usable. LUMI and Flatiron jobs must pass their on-site qualifications
before using full-sweep results in a hardware comparison.
