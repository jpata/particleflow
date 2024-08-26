#!/bin/bash
#SBATCH --partition main
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 6G
#SBATCH -o slurm-%x-%j-%N.out
set -e
set -x

OUTDIR=/local/joosep/mlpf/cms/v3/nopu/
CMSSWDIR=/home/joosep/CMSSW_12_3_0_pre6
MLPF_PATH=/home/joosep/particleflow/

#seed must be greater than 0
SAMPLE=$1
SEED=$2

WORKDIR=/scratch/local/joosep/$SLURM_JOBID/$SAMPLE/$SEED
#WORKDIR=`pwd`/$SAMPLE/$SEED
mkdir -p $WORKDIR
mkdir -p $OUTDIR

PILEUP=NoPileUp
PILEUP_INPUT=

N=100

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
  --conditions auto:phase1_2021_realistic \
  -n $N \
  --era Run3 \
  --eventcontent FEVTDEBUGHLT \
  -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT \
  --datatier GEN-SIM \
  --geometry DB:Extended \
  --pileup $PILEUP \
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
cmsRun step2_phase1_new.py > /dev/null
cmsRun step3_phase1_new.py > /dev/null
#cmsRun $CMSSWDIR/src/Validation/RecoParticleFlow/test/pfanalysis_ntuple.py
mv pfntuple.root pfntuple_${SEED}.root
python3 ${MLPF_PATH}/mlpf/data_cms/postprocessing2.py --input pfntuple_${SEED}.root --outpath ./ --save-normalized-table
bzip2 -z pfntuple_${SEED}.pkl
cp *.pkl.bz2 $OUTDIR/$SAMPLE/raw/
#cp pfntuple_${SEED}.root $OUTDIR/$SAMPLE/root/
rm -Rf $WORKDIR
