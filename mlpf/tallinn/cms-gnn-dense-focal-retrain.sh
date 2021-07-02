#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 5
#SBATCH --mem-per-gpu=8G

IMG=/home/software/singularity/base.simg:latest
cd ~/particleflow

#TF training
singularity exec --nv $IMG python3 mlpf/launcher.py --model-spec parameters/cms-gnn-dense-focal.yaml --action train --modifier retrain_energy --recreate --weights experiments/cms-gnn-dense-focal-285ae825.gpu0.local/weights-300-1.175282.hdf5
