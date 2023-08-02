#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out
set -e
set -x

env
df -h

WORKDIR=/scratch/$USER/${SLURM_JOB_ID}
SAMPLE=$1
SEED=$2
OUTDIR=/local/joosep/mlpf/gen/v3/

mkdir -p $WORKDIR
cd $WORKDIR

time /home/joosep/particleflow/mlpf/data_cms/genjob.sh $SAMPLE $SEED

#cp $WORKDIR/$SAMPLE/$SEED/pfntuple_*.root $OUTDIR/$SAMPLE/root/
cp $WORKDIR/$SAMPLE/$SEED/pfntuple_*.pkl.bz2 $OUTDIR/$SAMPLE/raw/

rm -Rf $WORKDIR
