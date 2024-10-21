#!/bin/bash
#SBATCH --partition main
#SBATCH --mem-per-cpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

export KERAS_BACKEND=tensorflow
export PYTHONPATH="mlpf"
export IMG=/home/software/singularity/pytorch.simg:2024-08-18
export CMD="singularity exec -B /local -B /scratch/persistent $IMG tfds build"
export DATA_DIR=/local/joosep/mlpf/tensorflow_datasets/cms
export MANUAL_DIR=/local/joosep/mlpf/cms/20240823_simcluster

$CMD mlpf/heptfds/cms_pf/$1 --config $2 --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/$3 --overwrite
