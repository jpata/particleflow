#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 8
#SBATCH --mem-per-gpu=8G

#IMG=/home/software/singularity/base.simg:latest
IMG=/home/joosep/singularity/tf26.simg
cd ~/particleflow

#TF training
singularity exec --nv $IMG python3 mlpf/pipeline.py train -c parameters/cms.yaml --plot-freq 1
