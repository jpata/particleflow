#!/bin/bash
set -e
set -x

OUTDIR=/local/joosep/mlpf/cms/20251125_cmssw_15_0_5_117d32/pu55to75_val/
CMSSWDIR=/scratch/persistent/joosep/CMSSW_15_0_5/
MLPF_PATH=/home/joosep/particleflow/

#seed must be greater than 0
SAMPLE=$1
SEED=$2

WORKDIR=/scratch/local/joosep/$SLURM_JOBID/$SAMPLE/$SEED
#WORKDIR=`pwd`/$SAMPLE/$SEED
mkdir -p $WORKDIR
mkdir -p $OUTDIR/$SAMPLE/root

PILEUP=Run3_Flat55To75_PoissonOOTPU
PILEUP_INPUT=filelist:${MLPF_PATH}/mlpf/data/cms/pu_files_local_val2.txt

N=50

env
source /cvmfs/cms.cern.ch/cmsset_default.sh

cd $CMSSWDIR
eval `scramv1 runtime -sh`
which python
which python3

env

cd $WORKDIR

#Generate the MC
cmsDriver.py $SAMPLE \
  --conditions 140X_mcRun3_2024_realistic_v26 \
  --beamspot DBrealistic \
  -n $N \
  --era Run3_2024 \
  --eventcontent FEVTDEBUGHLT \
  -s GEN,SIM,DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2023 \
  --datatier GEN-SIM \
  --geometry DB:Extended \
  --pileup $PILEUP \
  --pileup_input $PILEUP_INPUT \
  --no_exec \
  --fileout step2_phase1_new.root \
  --customise Validation/RecoParticleFlow/customize_pfanalysis.customize_step2 \
  --python_filename=step2_phase1_new.py

#  --customise_commands "process.mix.input.nbPileupEvents.probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120) \n process.mix.input.nbPileupEvents.probValue = cms.vdouble(0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446,0.00826446)"

#Run the reco sequences
cmsDriver.py step3 \
  --conditions 140X_mcRun3_2024_realistic_v26 \
  --beamspot DBrealistic \
  --era Run3_2024 \
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
cmsRun step2_phase1_new.py > /dev/null
cp step2_phase1_new.root $OUTDIR/$SAMPLE/root/step2_${SEED}.root

cmsRun step3_phase1_new.py > /dev/null
cp pfntuple.root $OUTDIR/$SAMPLE/root/pfntuple_${SEED}.root

rm -Rf /scratch/local/joosep/$SLURM_JOBID
