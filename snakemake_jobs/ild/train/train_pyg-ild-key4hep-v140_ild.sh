#!/bin/bash
set -e
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/scratch/local/joosep/tmp/job_${SLURM_JOB_ID:-manual-$$}"
export TEMPDIR=$TMPDIR
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR
cd /home/joosep/particleflow

export PYTHONPATH=$(pwd):$PYTHONPATH
export TFDS_DATA_DIR=/local/joosep/mlpf/ild/v1.4.0/50k/tfds
export KERAS_BACKEND=torch
export TORCH_COMPILE_DISABLE=1
env
nvidia-smi
python3 mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-ild-key4hep-v140 --production-name ild train --gpus 1
