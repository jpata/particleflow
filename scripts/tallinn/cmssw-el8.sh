#!/bin/bash
#SBATCH -p main
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH -o logs/slurm-%x-%j-%N.out

source /cvmfs/cms.cern.ch/cmsset_default.sh
export UNPACKED_IMAGE=/cvmfs/singularity.opensciencegrid.org/cmssw/cms\:rhel8-x86_64/
cmssw-el8 -B /root -B /cms -B /local -B /scratch/persistent -B /scratch/local --command-to-run $@
