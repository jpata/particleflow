# CLD batch-sweep site setup

Use the [portable benchmark workflow](benchmark-portability.md) for the shared
recipe, submission, data preflight, evidence and interpretation. This page covers
LUMI and Flatiron environment setup; [EmpireAI setup](../../scripts/empireai/README.md)
covers ARM64 GH200. Site workers and historical recipes remain available for
reproducing old runs, but new comparisons use the generic profile-driven launcher.

## Source and data

Clone merged `main`, then record and use the same commit on every platform:

```bash
git clone --branch main https://github.com/jpata/particleflow.git
cd particleflow
git rev-parse HEAD
mkdir -p logs_slurm
```

Use `git switch --detach COMMIT` in each checkout to freeze that revision.
Environments are architecture-specific; do not copy an ARM virtual environment
to an x86 site. The former benchmark development branch and incremental Git
bundle are no longer required.

Set `MLPF_CLD_DATA_DIR` to a compute-visible TFDS root containing
`cld_edm_ttbar_hits/1/3.2.2/`. The reference copy has 90,000 train and 10,000 test
events, 1,155 files and 31,326,294,015 bytes. Transfer the complete config `1`,
including train/test shards and metadata, using your site's normal data-transfer
method. The versioned inventory is
[`cld_ttbar_hits_config1_3.2.2.json`](../../configs/training/data/cld_ttbar_hits_config1_3.2.2.json).
Matching a TFDS version label alone does not establish identical data.

## LUMI: one MI250X GCD

[`lumi_mi250x_single.yaml`](../../configs/training/platforms/lumi_mi250x_single.yaml)
requests one GCD on `small-g`, seven CPU cores and 60G host RAM. One GCD has
64 GB HBM; it is half of a physical MI250X module. Set the project, immutable
LAIF image and overlay before setup and execution:

```bash
export LUMI_PROJECT_ROOT=/scratch/project_465001293
export MLPF_CLD_DATA_DIR="$LUMI_PROJECT_ROOT/$USER/mlpf-benchmark/tfds"
export IMG=$(readlink -f /appl/local/laifs/containers/lumi-multitorch-latest.sif)
export LUMI_VENV="$PWD/particleflow-laif-env"
bash scripts/lumi/setup_env.sh
bash scripts/lumi/batch_python.sh -m pip freeze > logs_slurm/lumi-pip-freeze.txt
bash scripts/lumi/batch_python.sh -c '
import mlpf.pipeline
import torch
from mlpf.model.set_decoder import flash_attn_varlen_func
assert torch.version.hip is not None
assert flash_attn_varlen_func is not None
print(torch.__version__, torch.version.hip)
'
bash scripts/lumi/batch_python.sh scripts/training/check_batch_data.py "$MLPF_CLD_DATA_DIR"
bash scripts/lumi/batch_python.sh -m scripts.training.batch_sweep \
  --platform configs/training/platforms/lumi_mi250x_single.yaml --dry-run
```

The overlay preserves the LAIF PyTorch/ROCm/FlashAttention stack. Keep the same
resolved `IMG` for setup and jobs; recreate the overlay when changing images.
Do not install the root project's NVIDIA wheel sources into it. Consult the
[LAIF environment guide](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/)
for site details.

For submission, use `scripts/lumi/batch_python.sh` as the environment adapter in
the [shared qualification command](benchmark-portability.md#qualification).
The profile uses two loader workers and `MLPF_SHARING_STRATEGY=file_descriptor`.
The adapter preserves Slurm's GPU visibility. Change the account/profile and
`LUMI_PROJECT_ROOT` together when using another project.

## Flatiron: one H100 PCIe

[`flatiron_h100_benchmark.yaml`](../../configs/training/platforms/flatiron_h100_benchmark.yaml)
requests one H100, 24 CPU cores and 128G host RAM. Prepare the CUDA environment
once on the appropriate site host:

```bash
export MLPF_CLD_DATA_DIR=/mnt/ceph/users/$USER/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds
module --force purge
module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 nccl openmpi apptainer
uv sync
uv pip freeze --python .venv/bin/python3 > logs_slurm/h100-pip-freeze.txt
bash scripts/flatiron/batch_python.sh -c 'import mlpf.pipeline; import torch; print(torch.__version__, torch.version.cuda)'
bash scripts/flatiron/batch_python.sh scripts/training/check_batch_data.py "$MLPF_CLD_DATA_DIR"
bash scripts/flatiron/batch_python.sh -m scripts.training.batch_sweep \
  --platform configs/training/platforms/flatiron_h100_benchmark.yaml --dry-run
```

Use `scripts/flatiron/batch_python.sh` as the environment adapter. It uses
`uv run --no-sync`, so case processes do not resolve dependencies repeatedly.
Leave cuDNN to the PyTorch wheel; the older site `cudnn/9.2` module conflicts with
its bundled libraries. Check the profile's partition, constraint and account
against your allocation before submission.

## Historical runs and current qualification

The [published v3.2.0 evidence and limitations](benchmark-portability.md#historical-evidence-and-remaining-limitations)
record the original GH200/H100/MI250X campaign. H100 used TFDS 3.2.1 while
GH200/LUMI used 3.2.2; source trees, attention backends and loader settings also
differed. The frozen `cld_h100_backbone_benchmark_321` and
`cld_gh200_backbone_benchmark` recipes describe those runs. New runs default to
`cld_backbone_benchmark`.

Local shell tests use stubs for Slurm, modules and containers. They verify argument
forwarding and visibility preservation, not remote runtime availability. Qualify
both backbones on each site using the final source commit before a full sweep.
