#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 5
#SBATCH --mem-per-gpu=8G

#IMG=/home/software/singularity/base.simg:latest
IMG=~/HEP-KBFI/singularity/base.simg
cd ~/particleflow/delphes

#TF training
singularity exec --nv $IMG python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/delphes-transformer-skipconn.yaml
