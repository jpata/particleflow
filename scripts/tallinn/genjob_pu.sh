#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out

env
df -h

WORKDIR=/scratch/$USER/${SLURM_JOB_ID}
SAMPLE=$1
SEED=$2
OUTDIR=/local/joosep/mlpf/gen/v3/

mkdir -p $WORKDIR
cd $WORKDIR

/home/joosep/particleflow/mlpf/data_cms/genjob_pu.sh $SAMPLE $SEED

#cp $WORKDIR/$SAMPLE/$SEED/pfntuple_*.root $OUTDIR/$SAMPLE/root/
cp $WORKDIR/$SAMPLE/$SEED/pfntuple_*.pkl.bz2 $OUTDIR/$SAMPLE/raw/

rm -Rf $WORKDIR
