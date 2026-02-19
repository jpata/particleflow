#!/bin/bash
set -e
set -x
unset PYTHONPATH

# Parameters
SAMPLE=$1
SEED=$2
JOB_TYPE=$3
USE_CUDA=${4:-false}

# Environment variables (usually set by Snakemake or spec)
CMSSWDIR=${CMSSWDIR:-/scratch/persistent/joosep/CMSSW_15_0_5/}
CONFIG_DIR=${CONFIG_DIR:-/home/joosep/particleflow/}
WORKSPACE_DIR=${WORKSPACE_DIR}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-pu55to75_val}

# Derived paths
INPUT_DIR=${WORKSPACE_DIR}/gen/${OUTPUT_SUBDIR}/${SAMPLE}/root
STEP2_FILE=${INPUT_DIR}/step2_${SEED}.root
OUTDIR=${WORKSPACE_DIR}/val/${JOB_TYPE}/${OUTPUT_SUBDIR}/${SAMPLE}

if [ ! -f "$STEP2_FILE" ]; then
    echo "Error: Input file $STEP2_FILE not found."
    exit 1
fi

if [ -z "$WORKDIR" ]; then
    WORKDIR=/scratch/local/joosep/val_${SAMPLE}_${SEED}_${JOB_TYPE}
fi

mkdir -p $WORKDIR
mkdir -p $OUTDIR

cleanup() {
    if [ ! -z "$WORKDIR" ] && [ "$WORKDIR" != "/scratch/local/joosep" ]; then
        echo "Cleaning up scratch directory $WORKDIR"
        rm -Rf $WORKDIR
    fi
}
trap cleanup EXIT

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd $CMSSWDIR
eval `scramv1 runtime -sh`

cd $WORKDIR

# Re-run reco sequences
CONDITIONS=140X_mcRun3_2024_realistic_v26
ERA=Run3
GEOM=DB:Extended

if [ "$JOB_TYPE" == "mlpf" ]; then
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT \
        --datatier RECOSIM,MINIAODSIM -n -1 --era $ERA \
        --eventcontent RECOSIM,MINIAODSIM --geometry=$GEOM \
        --filein file:$STEP2_FILE --fileout file:step3.root \
        --procModifiers mlpf --no_exec --python_filename=step3.py
    echo "process.mlpfProducer.use_cuda = ${USE_CUDA}" >> step3.py
else
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT \
        --datatier RECOSIM,MINIAODSIM -n -1 --era $ERA \
        --eventcontent RECOSIM,MINIAODSIM --geometry=$GEOM \
        --filein file:$STEP2_FILE --fileout file:step3.root \
        --no_exec --python_filename=step3.py
fi

cmsRun step3.py

# JME NANO recipe
cmsDriver.py step4_jme -s NANO:@JME --mc --conditions $CONDITIONS --era $ERA \
    --eventcontent NANOAODSIM --datatier NANOAODSIM \
    --customise_commands="process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)));process.MessageLogger.cerr.FwkReport.reportEvery=1000" \
    -n -1 --no_exec --filein file:step3_inMINIAODSIM.root --fileout file:step4_NANO_jme.root --python_filename=step4_jme.py

cmsRun step4_jme.py

# Copy outputs
cp step4_NANO_jme.root ${OUTDIR}/step4_NANO_jme_${SEED}.root
