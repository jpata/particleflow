#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=8G

IMG=/home/software/singularity/base.simg:latest
cd ~/particleflow

#TF training
singularity exec --nv $IMG python3 mlpf/launcher.py --model-spec parameters/cms-transformer-skipconn-gun.yaml --action train
