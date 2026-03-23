#!/bin/bash

module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer

export PYTHONPATH=`pwd`
source ~/miniforge3/bin/activate mlpf
which python3
python3 --version
