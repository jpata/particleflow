#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out

NJOB=$1

#change this as needed
OUTDIR=/home/joosep/particleflow/data

INPUT_FILELIST=$CMSSW_BASE/src/Validation/RecoParticleFlow/test/tmp/das_cache/QCD_PU.txt

set -e
set -v
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/grid.cern.ch/c7ui-test/etc/profile.d/setup-c7-ui-example.sh

cd $CMSSW_BASE

eval `scramv1 runtime -sh`

CONDITIONS=auto:phase1_2021_realistic ERA=Run3 GEOM=DB.Extended CUSTOM=
FILENAME=`sed -n "${NJOB}p" $INPUT_FILELIST`
NTHREADS=1

WORKDIR=/scratch/$USER/job_${SLURM_JOB_ID}_${NJOB}
mkdir -p $WORKDIR
cd $WORKDIR

cmsDriver.py step3 --conditions $CONDITIONS -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@nanoAODDQM --datatier RECOSIM,MINIAODSIM,DQMIO --nThreads 1 -n -1 --era $ERA --eventcontent RECOSIM,MINIAODSIM,DQM --geometry=$GEOM --filein $FILENAME --fileout file:step3.root
ls *.root

mkdir -p $OUTDIR/QCDPU_baseline/
cp step3_inMINIAODSIM.root $OUTDIR/QCDPU_baseline/step3_MINI_${NJOB}.root

rm -Rf $WORKDIR
