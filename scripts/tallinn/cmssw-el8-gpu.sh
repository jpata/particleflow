#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out

source /cvmfs/cms.cern.ch/cmsset_default.sh
export UNPACKED_IMAGE=/cvmfs/singularity.opensciencegrid.org/cmssw/cms\:rhel8-x86_64/
cmssw-el8 --env X509_USER_PROXY=/home/joosep/x509 --nv -B /scratch/persistent -B /scratch/local --command-to-run $@
