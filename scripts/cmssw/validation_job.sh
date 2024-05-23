#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out

JOBTYPE=$1
INPUT_FILELIST=$2
SAMPLE=$3
NJOB=$4

#change this as needed, need enough space for outputs
OUTDIR=$CMSSW_BASE/out/
WORKDIR=$CMSSW_BASE/work_${SAMPLE}_${JOBTYPE}_${NJOB}

#when running at T2_EE_Estonia
# OUTDIR=/local/joosep/mlpf/results/cms/${CMSSW_VERSION}/
# WORKDIR=/scratch/local/$USER/${SLURM_JOB_ID}

#abort on error, print all commands
set -e
set -x

CONDITIONS=auto:phase1_2021_realistic ERA=Run3 GEOM=DB.Extended CUSTOM=
FILENAME=`sed -n "${NJOB}p" $INPUT_FILELIST`
NTHREADS=1

mkdir -p $WORKDIR
cd $WORKDIR

if [ $JOBTYPE == "mlpf" ]; then
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@nanoAODDQM \
	--datatier RECOSIM,MINIAODSIM,DQMIO --nThreads 1 -n -1 --era $ERA \
	--eventcontent RECOSIM,MINIAODSIM,DQM --geometry=$GEOM \
	--filein $FILENAME --fileout file:step3.root --procModifiers mlpf
elif [ $JOBTYPE == "pf" ]; then
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@nanoAODDQM \
	--datatier RECOSIM,MINIAODSIM,DQMIO --nThreads 1 -n -1 --era $ERA \
	--eventcontent RECOSIM,MINIAODSIM,DQM --geometry=$GEOM \
	--filein $FILENAME --fileout file:step3.root
fi
ls *.root

mkdir -p $OUTDIR/${SAMPLE}_${JOBTYPE}
cp step3_inMINIAODSIM.root $OUTDIR/${SAMPLE}_${JOBTYPE}/step3_MINI_${NJOB}.root

rm -Rf $WORKDIR
