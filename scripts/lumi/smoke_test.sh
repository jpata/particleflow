#!/bin/bash
#SBATCH --account=project_465001293
#SBATCH --partition=small-g
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs_slurm/laif-smoke-%j.out
#SBATCH --error=logs_slurm/laif-smoke-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export PYTHONNOUSERSITE=1
export IMG=${IMG:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}
export LUMI_VENV=${LUMI_VENV:-$PWD/particleflow-laif-env}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export NCCL_DEBUG=INFO
srun singularity run -B /scratch/project_465001293 -B /tmp "$IMG" \
  bash -c 'set -e; source "$LUMI_VENV/bin/activate"; exec python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/lumi/smoke_test.py'
