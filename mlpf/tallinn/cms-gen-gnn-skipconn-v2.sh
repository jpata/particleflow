#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 5
#SBATCH --mem-per-gpu=8G

IMG=/home/software/singularity/base.simg:latest
cd ~/particleflow

#TF training
singularity exec -B /home -B /scratch-persistent --nv $IMG python3 mlpf/launcher.py --model-spec parameters/cms-gen-gnn-skipconn-v2.yaml --action train
