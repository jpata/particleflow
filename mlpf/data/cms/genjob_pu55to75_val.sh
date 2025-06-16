#!/bin/bash
set -e
set -x

OUTDIR=/local/joosep/mlpf/cms/20250612_cmssw_15_0_5_f8ae2f/pu55to75_val/
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
PILEUP_INPUT=filelist:${MLPF_PATH}/mlpf/data/cms/pu_files_local.txt

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
  --conditions auto:phase1_2025_realistic \
  --beamspot Realistic2024ppRefCollision \
  -n $N \
  --era Run3_2025 \
  --eventcontent FEVTDEBUGHLT \
  -s GEN,SIM,DIGI:pdigi_valid,L1,DIGI2RAW,HLT \
  --datatier GEN-SIM-DIGI-RAW \
  --geometry DB:Extended \
  --pileup $PILEUP \
  --pileup_input $PILEUP_INPUT \
  --no_exec \
  --fileout step2_phase1_new.root \
  --python_filename=step2_phase1_new.py

pwd
ls -lrt

echo "process.RandomNumberGeneratorService.generator.initialSeed = $SEED" >> step2_phase1_new.py
cmsRun step2_phase1_new.py > /dev/null
cp step2_phase1_new.root $OUTDIR/$SAMPLE/root/step2_${SEED}.root

rm -Rf /scratch/local/joosep/$SLURM_JOBID
