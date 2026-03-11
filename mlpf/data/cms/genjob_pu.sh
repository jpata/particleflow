#!/bin/bash
set -e
set -x
unset PYTHONPATH

#seed must be greater than 0
SAMPLE=$1
SEED=$2
PU_TYPE=${3:-nopu}
COPY_STEP2=${4:-False}

CMSSWDIR=${CMSSWDIR:-/scratch/persistent/joosep/CMSSW_15_0_5/}
CONFIG_DIR=${CONFIG_DIR:-/home/joosep/particleflow/}
OUTDIR_DEFAULT_BASE=/local/joosep/mlpf/cms/20250508_cmssw_15_0_5_d3c6d1

if [ "$PU_TYPE" == "nopu" ]; then
    PILEUP=NoPileUp
    PILEUP_INPUT=
    NEV_DEFAULT=100
    OUTDIR_SUFFIX="nopu"
elif [ "$PU_TYPE" == "pu55to75" ]; then
    PILEUP=Run3_Flat55To75_PoissonOOTPU
    #classical mixing for Summer24 /MinBias_TuneCP5_13p6TeV-pythia8/RunIII2024Summer24GS-140X_mcRun3_2024_realistic_v20-v1/GEN-SIM
    PILEUP_INPUT=filelist:${CONFIG_DIR}/mlpf/data/cms/pu_files_local.txt
    NEV_DEFAULT=50
    OUTDIR_SUFFIX="pu55to75"
else
    echo "Unknown PU_TYPE: $PU_TYPE"
    exit 1
fi

OUTDIR=${OUTDIR:-${OUTDIR_DEFAULT_BASE}/${OUTDIR_SUFFIX}/}
N=${NEV:-$NEV_DEFAULT}

if [ -z "$WORKDIR" ]; then
    if [ ! -z "$SLURM_JOBID" ]; then
        WORKDIR=/scratch/local/joosep/$SLURM_JOBID/$SAMPLE/$SEED
        CLEANUP_DIR=/scratch/local/joosep/$SLURM_JOBID
    else
        WORKDIR=/scratch/local/joosep/job_${SAMPLE}_${SEED}
        CLEANUP_DIR=$WORKDIR
    fi
else
    CLEANUP_DIR=$WORKDIR
fi

mkdir -p $WORKDIR
mkdir -p $OUTDIR/$SAMPLE/root/

# Ensure cleanup on exit, even if the job fails
cleanup() {
    # Safety check: never delete the root scratch directory
    if [ ! -z "$CLEANUP_DIR" ] && [ "$CLEANUP_DIR" != "/scratch/local/joosep" ] && [ "$CLEANUP_DIR" != "/scratch/local/joosep/" ]; then
        echo "Cleaning up scratch directory $CLEANUP_DIR"
        rm -Rf $CLEANUP_DIR
    fi
}
trap cleanup EXIT

env
source /cvmfs/cms.cern.ch/cmsset_default.sh

cd $CMSSWDIR
eval `scramv1 runtime -sh`

env

cd $WORKDIR

#PU arguments
PU_ARGS=("--pileup" "$PILEUP")
if [ ! -z "$PILEUP_INPUT" ]; then
    PU_ARGS+=("--pileup_input" "$PILEUP_INPUT")
fi

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
  "${PU_ARGS[@]}" \
  --no_exec \
  --fileout step2_phase1_new.root \
  --customise Validation/RecoParticleFlow/customize_pfanalysis.customize_step2 \
  --python_filename=step2_phase1_new.py

#Run the reco sequences
cmsDriver.py step3 \
  --conditions 140X_mcRun3_2024_realistic_v26 \
  --beamspot DBrealistic \
  -n -1 \
  --era Run3_2024 \
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
if [ "$COPY_STEP2" == "True" ]; then
  cp step2_phase1_new.root $OUTDIR/$SAMPLE/root/step2_${SEED}.root
fi

cmsRun step3_phase1_new.py > /dev/null
#cp step3_phase1_new.root $OUTDIR/$SAMPLE/root/step3_${SEED}.root
mv pfntuple.root pfntuple_${SEED}.root
cp pfntuple_${SEED}.root $OUTDIR/$SAMPLE/root/
