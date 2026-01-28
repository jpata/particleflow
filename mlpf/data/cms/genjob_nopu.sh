#!/bin/bash
set -e
set -x

OUTDIR=${OUTDIR:-/local/joosep/mlpf/cms/20250508_cmssw_15_0_5_d3c6d1/nopu/}
CMSSWDIR=${CMSSWDIR:-/scratch/persistent/joosep/CMSSW_15_0_5/}
CONFIG_DIR=${CONFIG_DIR:-/home/joosep/particleflow/}

#seed must be greater than 0
SAMPLE=$1
SEED=$2

WORKDIR=${WORKDIR:-/scratch/local/joosep/$SLURM_JOBID/$SAMPLE/$SEED}
#WORKDIR=`pwd`/$SAMPLE/$SEED
mkdir -p $WORKDIR
mkdir -p $OUTDIR

PILEUP=NoPileUp
PILEUP_INPUT=

N=${NEV:-100}

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
  --conditions auto:phase1_2023_realistic \
  --beamspot Realistic25ns13p6TeVEarly2023Collision \
  -n $N \
  --era Run3_2023 \
  --eventcontent FEVTDEBUGHLT \
  -s GEN,SIM,DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2023 \
  --datatier GEN-SIM \
  --geometry DB:Extended \
  --pileup $PILEUP \
  --no_exec \
  --fileout step2_phase1_new.root \
  --customise Validation/RecoParticleFlow/customize_pfanalysis.customize_step2 \
  --python_filename=step2_phase1_new.py

#Run the reco sequences
cmsDriver.py step3 \
  --conditions auto:phase1_2023_realistic \
  --beamspot Realistic25ns13p6TeVEarly2023Collision \
  --era Run3_2023 \
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
#cp step2_phase1_new.root $OUTDIR/$SAMPLE/root/step2_${SEED}.root

cmsRun step3_phase1_new.py > /dev/null
#cp step3_phase1_new.root $OUTDIR/$SAMPLE/root/step3_${SEED}.root
mv pfntuple.root pfntuple_${SEED}.root
cp pfntuple_${SEED}.root $OUTDIR/$SAMPLE/root/

# python3 ${MLPF_PATH}/mlpf/data/cms/postprocessing2.py --input pfntuple_${SEED}.root --outpath ./
# bzip2 -z pfntuple_${SEED}.pkl
# cp *.pkl.bz2 $OUTDIR/$SAMPLE/raw/

rm -Rf /scratch/local/joosep/$SLURM_JOBID
