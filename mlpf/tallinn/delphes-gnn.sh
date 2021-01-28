#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=8G

IMG=/home/software/singularity/base.simg:latest
cd ~/particleflow/delphes

#TF training
singularity exec --nv $IMG python3 ../mlpf/tensorflow/delphes_model.py parameters/delphes-gnn.yaml
