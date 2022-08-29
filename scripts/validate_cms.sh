#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh

env
df -h

WORKDIR=/scratch/$USER/${SLURM_JOB_ID}
SAMPLE=$1
SEED=$2

mkdir -p $WORKDIR
cd $WORKDIR
