# EmpireAI Grace Hopper setup

Use the [portable benchmark workflow](../../docs/training/benchmark-portability.md)
for shared sweep submission, recorder artifacts and evidence export. The
[site runbook](../../docs/training/hpc-batch-sweep.md) covers LUMI and Flatiron.

## ARM allocation and storage

[`empireai_gh200.yaml`](../../configs/training/platforms/empireai_gh200.yaml)
selects one GH200 with account `cuny`, QoS `burst`, partition `grace`, constraint
`gracehopper` and reservation `ghtest`. The recorded reservation expiry is
2026-11-27; verify availability and update the profile when the allocation changes.

The 2026-10-29 `alphagh01` probe found ARM64, 72 CPU cores, 64 KiB pages, 97,871 MiB
GPU memory and driver 595.45.04. That node mounted `/mnt/home` and `/projects`,
but not `/mnt/lustre` or `/ddn`. Use compute-visible storage for training data.
The [EmpireAI storage guide](https://empireai.freshdesk.com/en/support/solutions/articles/157000376738-empire-ai-storage-architecture-user-workflow-guide)
recommends `/projects`; the old Lustre path is a transfer source on the login node.

## Install the environment

From the repository on the login node:

```bash
srun --partition=grace --account=cuny --qos=burst --reservation=ghtest \
  --constraint=gracehopper --cpus-per-task=8 --mem=32G --time=00:30:00 \
  bash scripts/empireai/setup_env.sh
```

Setup uses the isolated lock in `envs/empireai-gh200/` and installs its ARM64
Python 3.11 environment. It imports the repository through `PYTHONPATH`, avoiding
the root project's x86 dependency sources. Do not run root `uv run` here or reuse
this environment on an x86 login node.

Attention and HEPTv2 use PyTorch SDPA/FlexAttention and native packed varlen
set-decoder attention with zero dropout. External `flash-attn`, `torch-scatter`,
LitePT/spconv, data-production tools and Ray HPO are not installed.

## Prepare data and run the smoke recipe

The smoke recipe needs the complete `cld_edm_ttbar_hits/1/3.2.2` train/test copy,
about 30 GiB. From the login node with source Lustre access:

```bash
mkdir -p data/empireai-smoke/tfds/cld_edm_ttbar_hits
rsync -a --stats \
  /mnt/lustre/cuny/jpata/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds/cld_edm_ttbar_hits/1 \
  data/empireai-smoke/tfds/cld_edm_ttbar_hits/
```

This repository-local copy is for qualification and counts against home quota.
Export `MLPF_CLD_DATA_DIR` before preview/submission to use another TFDS root.
Source `scripts/empireai/environment.sh` for direct Python commands in an ARM
allocation; it configures the interpreter, data default and repository imports.

Preview the smoke submission from an ARM allocation:

```bash
srun --partition=grace --account=cuny --qos=burst --reservation=ghtest \
  --constraint=gracehopper --cpus-per-task=2 --mem=8G --time=00:10:00 \
  bash scripts/empireai/train_scenario.sh cld_gh200_smoke gh200 --dry-run
```

Remove `--dry-run` to submit: 20 steps at global batch 2, validation/checkpoints
every 10 steps, four validation/test events. Outputs are in
`experiments/cld_gh200_smoke/` and logs in `logs_slurm/`. Use `--continue` to resume
an interrupted compatible run; a completed run is skipped. For direct execution
inside a GPU allocation:

```bash
bash scripts/empireai/run_scenario.sh \
  configs/training/scenarios/cld_gh200_smoke.yaml \
  configs/training/platforms/empireai_gh200.yaml
```

For a batch sweep, use the shared submitter with profile
`configs/training/platforms/empireai_gh200.yaml` and adapter
`scripts/empireai/batch_python.sh`, from the prepared ARM environment.

## Utilization evidence for an allocation request

Recorder files, units, aggregation and timing limits are documented once in the
[shared metric reference](../../docs/training/benchmark-portability.md#recorder-artifacts-and-metric-scope).
Set `--set benchmark.enabled=true` for scenario recording; the sweep enables it
explicitly. Use `python3 -m mlpf.benchmark_report RUNS_DIRECTORY` to export it.

The [pinned historical package](../../docs/training/benchmark-portability.md#historical-evidence-and-remaining-limitations)
contains the original cross-platform plots. The local
[`gh200_batch_sweep_153260.json`](../../configs/training/references/gh200_batch_sweep_153260.json)
retains the GH200 reference. The frozen `cld_gh200_backbone_benchmark` recipe is the
1,000-step batch-8 study; new sweeps use `cld_backbone_benchmark`.
`analyze_utilization.py` and `summarize_benchmark.py` support the older TensorBoard
artifacts. Their sampled/window-mean throughput and utilization are not exact
whole-job averages. In particular, the older monitor did not retain window sample
counts or power, so energy cannot be reconstructed from those runs.

On 2026-10-29, smoke job `153187` completed forward/backward, validation, predictions
and checkpoint writes without external FlashAttention; a second run restored step
10 and completed step 20. Historical logs are `logs_slurm/gh200_smoke_resume.log`
and `logs_slurm/log_cld_gh200_smoke_153187_0.err`. These verify that recorded stack,
not the current availability of cluster modules, data or reservations.

For local shell tests in an ARM allocation, keep Lmod from replacing test stubs:

```bash
source scripts/empireai/environment.sh
unset -f module
export BASH_ENV=/dev/null
python3 -m pytest -q tests/test_training_submission.py tests/test_hpc_batch_sweep.py
```
