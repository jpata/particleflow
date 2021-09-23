#!/bin/bash
set -e
set -x

CMSSWDIR=/home/joosep/reco/mlpf/CMSSW_12_1_0_pre3
MLPF_PATH=/home/joosep/particleflow/

#seed must be greater than 0
SAMPLE=$1
SEED=$2

WORKDIR=`pwd`/$SAMPLE/$SEED
mkdir -p $WORKDIR

PILEUP=Run3_Flat55To75_PoissonOOTPU
PILEUP_INPUT=filelist:${MLPF_PATH}/mlpf/data/pu_files.txt

N=5

env
source /cvmfs/cms.cern.ch/cmsset_default.sh

cd $CMSSWDIR
eval `scramv1 runtime -sh`

cd $WORKDIR

#Generate the MC
cmsDriver.py $SAMPLE \
  --conditions auto:phase1_2021_realistic \
  -n $N \
  --era Run3 \
  --eventcontent FEVTDEBUGHLT \
  -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT \
  --datatier GEN-SIM \
  --geometry DB:Extended \
  --pileup $PILEUP \
  --pileup_input $PILEUP_INPUT \
  --no_exec \
  --fileout step2_phase1_new.root \
  --customise Validation/RecoParticleFlow/customize_pfanalysis.customize_step2 \
  --python_filename=step2_phase1_new.py

#Run the reco sequences
cmsDriver.py step3 \
  --conditions auto:phase1_2021_realistic \
  --era Run3 \
  -n -1 \
  --eventcontent FEVTDEBUGHLT \
  --runUnscheduled \
  -s RAW2DIGI,L1Reco,RECO,RECOSIM \
  --datatier GEN-SIM-RECO \
  --geometry DB:Extended \
  --no_exec \
  --filein file:step2_phase1_new.root \
  --fileout step3_phase1_new.root \
  --customise Validation/RecoParticleFlow/customize_pfanalysis.customize_step3 \
  --python_filename=step3_phase1_new.py

pwd
ls -lrt

echo "process.RandomNumberGeneratorService.generator.initialSeed = $SEED" >> step2_phase1_new.py
cmsRun step2_phase1_new.py >& step2.log
cmsRun step3_phase1_new.py >& step3.log
cmsRun $CMSSWDIR/src/Validation/RecoParticleFlow/test/pfanalysis_ntuple.py >& step4.log
mv pfntuple.root pfntuple_${SEED}.root
python3 ${MLPF_PATH}/mlpf/data/postprocessing2.py --input pfntuple_${SEED}.root --outpath ./ --save-normalized-table --events-per-file -1
rm step*.root
