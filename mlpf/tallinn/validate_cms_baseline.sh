#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out

NJOB=$1
INPUT_FILELIST=/home/joosep/reco/mlpf/CMSSW_12_3_0_pre6/src/Validation/RecoParticleFlow/test/tmp/das_cache/QCD_PU.txt

set -e
set -v
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/grid.cern.ch/c7ui-test/etc/profile.d/setup-c7-ui-example.sh

cd ~/reco/mlpf/CMSSW_12_3_0_pre6

eval `scramv1 runtime -sh`
export X509_USER_PROXY=/home/joosep/x509

CONDITIONS=auto:phase1_2021_realistic ERA=Run3 GEOM=DB.Extended CUSTOM=
FILENAME=`sed -n "${NJOB}p" $INPUT_FILELIST`
NTHREADS=1

WORKDIR=/scratch/$USER/${SLURM_JOB_ID}
mkdir -p $WORKDIR
cd $WORKDIR

cmsDriver.py step3 --conditions $CONDITIONS -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@nanoAODDQM --datatier RECOSIM,DQMIO --nThreads 1 -n -1 --era $ERA --eventcontent RECOSIM,DQM --geometry=$GEOM --filein $FILENAME --fileout file:step3.root
ls *.root

cmsDriver.py step4 --conditions $CONDITIONS -s HARVESTING:@standardValidation+@standardDQM+@ExtraHLT+@miniAODValidation+@miniAODDQM+@nanoAODDQM --era $ERA --filetype DQM --filein file:step3_inDQM.root --fileout file:step4.root
ls *.root

mkdir -p /home/joosep/particleflow/data/QCDPU_baseline/
cp DQM_*.root /home/joosep/particleflow/data/QCDPU_baseline/DQM_${NJOB}.root

rm -Rf $WORKDIR
