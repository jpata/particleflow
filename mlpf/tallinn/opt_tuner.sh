#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1

IMG=/home/software/singularity/base.simg:latest
cd ~/particleflow

#export SINGULARITYENV_KERASTUNER_TUNER_ID="tuner-${SLURM_JOB_ID}"
#export SINGULARITYENV_KERASTUNER_ORACLE_IP="127.0.0.1"
#export SINGULARITYENV_KERASTUNER_ORACLE_PORT="8000"

singularity exec -B /scratch --nv $IMG python3 mlpf/tensorflow/opt.py
