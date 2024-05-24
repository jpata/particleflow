#!/bin/bash
export JOBTYPE=$1
export INPUT_FILELIST=$2
export SAMPLE=$3
export NJOB=$4

#change this as needed, need enough space for outputs
# OUTDIR=$CMSSW_BASE/out/
# WORKDIR=$CMSSW_BASE/work_${SAMPLE}_${JOBTYPE}_${NJOB}

#when running at T2_EE_Estonia
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /scratch/persistent/joosep/CMSSW_14_1_0_pre3
eval `scram runtime -sh`
export OUTDIR=/local/joosep/mlpf/results/cms/${CMSSW_VERSION}/
export WORKDIR=/scratch/local/$USER/${SLURM_JOB_ID}

#abort on error, print all commands
set -e
set -x

export CONDITIONS=auto:phase1_2023_realistic ERA=Run3 GEOM=DB.Extended CUSTOM=
export FILENAME=`sed -n "${NJOB}p" $INPUT_FILELIST`
export NTHREADS=1

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
