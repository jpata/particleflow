#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 4
#SBATCH --mem-per-gpu=8G

IMG=/home/software/singularity/tf-2.8.0.simg
cd ~/particleflow

#TF training
singularity exec --nv --env PYTHONPATH=hep_tfds $IMG python mlpf/pipeline.py train -c parameters/cms-gen.yaml --plot-freq 10
