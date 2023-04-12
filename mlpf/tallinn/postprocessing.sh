#!/bin/bash
#SBATCH -p main
#SBATCH --cpus-per-task 30
#SBATCH --mem-per-cpu=1G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.11.0.simg
cd ~/particleflow

singularity exec -B /local -B /scratch/persistent --nv \
    $IMG python fcc/postprocessing_hits.py $1
