#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out

source /cvmfs/cms.cern.ch/cmsset_default.sh
export UNPACKED_IMAGE=/cvmfs/singularity.opensciencegrid.org/cmssw/cms\:rhel7-x86_64/
cmssw-el7 -B /root -B /cms -B /local -B /scratch/persistent -B /scratch/local --command-to-run $@
