#!/bin/bash
set -e
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TMPDIR=/scratch/local/joosep/tmp
export TEMPDIR=/scratch/local/joosep/tmp
export TEMP=/scratch/local/joosep/tmp
export TMP=/scratch/local/joosep/tmp
mkdir -p $TMPDIR
cd /home/joosep/particleflow2

export PYTHONPATH=$(pwd):$PYTHONPATH
export TFDS_DATA_DIR=/local/joosep/mlpf/clic/v1.2.5_key4hep_2025-05-29/tfds
export KERAS_BACKEND=torch
export TORCH_COMPILE_DISABLE=1
env
nvidia-smi
python3 mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-clic-v1 --production-name clic train --gpus 1
