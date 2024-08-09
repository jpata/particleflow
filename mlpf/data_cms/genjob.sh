#!/bin/bash
#SBATCH --partition main
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 6G
#SBATCH -o slurm-%x-%j-%N.out

scripts/tallinn/cmssw-el8.sh $@
