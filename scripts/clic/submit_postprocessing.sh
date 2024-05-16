#!/bin/bash
#SBATCH --partition main
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 2G
#SBATCH -o logs/slurm-%x-%j-%N.out

singularity exec -B /local /home/software/singularity/tf-2.14.0.simg python3.10 scripts/fcc/postprocessing.py $1
