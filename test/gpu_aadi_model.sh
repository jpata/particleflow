#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1


#Hello World
#singularity exec -B /scratch -B /home --nv /home/software/singularity/base.simg:2020-08-13 \
 # python3 hello_world.py


singularity exec -B /home --nv /home/software/singularity/base.simg:latest python3 aadi_model.py




~                           
