#!/bin/bash
#SBATCH --partition short
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 6G
#SBATCH -o slurm-%x-%j-%N.out
set -e
set -x

OUTDIR=/local/joosep/mlpf/cms/20240702_cptruthdef/nopu/
CMSSWDIR=/scratch/persistent/joosep/CMSSW_14_1_0_pre3
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

N=200

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
cmsRun step3_phase1_new.py > /dev/null
#cmsRun $CMSSWDIR/src/Validation/RecoParticleFlow/test/pfanalysis_ntuple.py
mv pfntuple.root pfntuple_${SEED}.root
python3 ${MLPF_PATH}/mlpf/data_cms/postprocessing2.py --input pfntuple_${SEED}.root --outpath ./
bzip2 -z pfntuple_${SEED}.pkl
cp *.pkl.bz2 $OUTDIR/$SAMPLE/raw/

#copy ROOT outputs
#cp step2_phase1_new.root $OUTDIR/$SAMPLE/root/step2_${SEED}.root
#cp step3_phase1_new.root $OUTDIR/$SAMPLE/root/step3_${SEED}.root
#cp pfntuple_${SEED}.root $OUTDIR/$SAMPLE/root/

rm -Rf $WORKDIR
