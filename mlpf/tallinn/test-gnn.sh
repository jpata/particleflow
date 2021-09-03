#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=8G

IMG=/home/software/singularity/base.simg:latest
cd ~/particleflow

#TF training
singularity exec --nv $IMG python3 mlpf/pipeline.py train -c parameters/test-gnn/cms-0l.yaml --plot-freq 10
singularity exec --nv $IMG python3 mlpf/pipeline.py train -c parameters/test-gnn/cms-lsh-1l.yaml --plot-freq 10
singularity exec --nv $IMG python3 mlpf/pipeline.py train -c parameters/test-gnn/cms-lsh-2l.yaml --plot-freq 10
singularity exec --nv $IMG python3 mlpf/pipeline.py train -c parameters/test-gnn/cms-lsh-3l.yaml --plot-freq 10
singularity exec --nv $IMG python3 mlpf/pipeline.py train -c parameters/test-gnn/cms-nolsh-1l.yaml --plot-freq 10
