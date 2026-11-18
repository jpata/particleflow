#!/bin/bash
#SBATCH --job-name=input_normalization_stats
#SBATCH --partition=scc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2:00:00
#SBATCH --output=logs_slurm/input_normalization_stats_%j.out

# Input normalization statistics of one dataset version over its whole train split
#
# Commands of the committed mlpf/model/input_normalization_stats/<dataset>_<version>.yaml
#   sbatch scripts/flatiron/input_normalization_stats.sh colliderml 1.5.0 /mnt/ceph/users/ewulff/tensorflow_datasets/colliderml --samples ttbar_nopu
#   sbatch scripts/flatiron/input_normalization_stats.sh cld 3.3.0 /mnt/ceph/users/jpata/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds --samples edm_qq edm_ttbar edm_ww_fullhad
#   sbatch scripts/flatiron/input_normalization_stats.sh clic 3.3.0 /mnt/ceph/users/jpata/mlpf/clic/v1.2.5_key4hep_2025-05-29/tfds
#   sbatch scripts/flatiron/input_normalization_stats.sh idea 0.1.0 /mnt/ceph/users/rmorel/DEPR_particle_physics/idea/tensorflow_datasets/idea

set -euo pipefail

DATASET=${1:?dataset is required}
VERSION=${2:?version is required}
DATA_DIR=${3:?TFDS data directory is required}
WORK_DIR=logs_slurm/input_normalization_stats_${SLURM_JOB_ID}

export PYTHONPATH=$PWD

srun .venv/bin/python scripts/compute_input_normalization_stats.py map "$DATASET" "$VERSION" --data-dir "$DATA_DIR" --work-dir "$WORK_DIR" "${@:4}"
.venv/bin/python scripts/compute_input_normalization_stats.py reduce "$DATASET" "$VERSION" --work-dir "$WORK_DIR"
