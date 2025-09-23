#!/bin/bash
#SBATCH --partition main
#SBATCH --mem-per-cpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

export KERAS_BACKEND=tensorflow
export PYTHONPATH="mlpf"
export IMG=/home/software/singularity/pytorch.simg:2025-09-01
export CMD="singularity exec -B /local -B /scratch/persistent -B /scratch/local $IMG tfds build"
export PYTHONUNBUFFERED=1

$CMD mlpf/heptfds/$1 --config $2 --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/$3 --overwrite
