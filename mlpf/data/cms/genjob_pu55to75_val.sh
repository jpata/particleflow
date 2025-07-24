#!/bin/bash
set -e
set -x

OUTDIR=/local/joosep/mlpf/cms/20250630_cmssw_15_0_5_f8ae2f/pu55to75_val/
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
PILEUP_INPUT=filelist:${MLPF_PATH}/mlpf/data/cms/pu_files_local_val.txt

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
  --conditions auto:phase1_2024_realistic \
  --beamspot Realistic2024ppRefCollision \
  -n $N \
  --era Run3_2024 \
  --eventcontent FEVTDEBUGHLT \
  -s GEN,SIM,DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2024 \
  --datatier GEN-SIM-DIGI-RAW \
  --geometry DB:Extended \
  --pileup $PILEUP \
  --pileup_input $PILEUP_INPUT \
  --no_exec \
  --fileout step2_phase1_new.root \
  --customise Validation/RecoParticleFlow/customize_pfanalysis.customize_step2 \
  --python_filename=step2_phase1_new.py

cmsDriver.py step3 \
  --conditions auto:phase1_2024_realistic \
  --beamspot Realistic2024ppRefCollision \
  --era Run3_2024 \
  -n -1 \
  --eventcontent FEVTDEBUGHLT \
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
#cp step3_phase1_new.root $OUTDIR/$SAMPLE/root/step3_${SEED}.root
cp pfntuple.root $OUTDIR/$SAMPLE/root/pfntuple_${SEED}.root

rm -Rf /scratch/local/joosep/$SLURM_JOBID
