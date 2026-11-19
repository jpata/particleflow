#!/bin/bash
set -e
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TMPDIR="/scratch/local/joosep/tmp/job_${SLURM_JOB_ID:-manual-$$}"
export TEMPDIR=$TMPDIR
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR
cd /home/joosep/particleflow

export PYTHONPATH=$(pwd):$PYTHONPATH
export TFDS_DATA_DIR=/local/joosep/mlpf/idea/v1.4.0/50k/tfds
export KERAS_BACKEND=torch
export TORCH_COMPILE_DISABLE=1
env
nvidia-smi
python3 mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-idea-pipeline-v1 --production-name idea train --gpus 1
