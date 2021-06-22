#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 5
#SBATCH --mem-per-gpu=8G

IMG=/home/software/singularity/base.simg:latest
cd ~/particleflow

#TF training
singularity exec --nv $IMG python3 mlpf/launcher.py --model-spec parameters/cms-gnn-skipconn.yaml --action train --weights experiments/cms-gnn-skipconn-9f17890f/weights-500-0.994515.hdf5
#CUDA_VISIBLE_DEVICES=0 singularity exec --nv $IMG python3 mlpf/launcher.py --model-spec parameters/cms-gnn-skipconn.yaml --action eval --weights experiments/cms-gnn-skipconn-6cfe8834/weights-328-1.010852.hdf5
