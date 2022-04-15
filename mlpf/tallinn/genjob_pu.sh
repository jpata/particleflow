#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=1

env
df -h

WORKDIR=/scratch/$USER/${SLURM_JOB_ID}
SAMPLE=$1
SEED=$2

mkdir -p $WORKDIR
cd $WORKDIR

/home/joosep/particleflow/mlpf/data/genjob_pu.sh $SAMPLE $SEED

#cp $WORKDIR/$SAMPLE/$SEED/pfntuple_*.root /hdfs/local/joosep/mlpf/gen/v2/$SAMPLE/root/ 
cp $WORKDIR/$SAMPLE/$SEED/pfntuple_*.pkl.bz2 /hdfs/local/joosep/mlpf/gen/v2/$SAMPLE/raw/

rm -Rf $WORKDIR
