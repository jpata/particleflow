#!/bin/bash
set -e
set -x
unset PYTHONPATH

# Parameters
SAMPLE=$1
SEED=$2 # SEED is used here as the line number in the input filelist
JOB_TYPE=$3
USE_CUDA=${4:-false}
NTHREADS=${NTHREADS:-8}

# Environment variables
CMSSWDIR=${CMSSWDIR:-/scratch/persistent/joosep/CMSSW_15_0_5/}
CONFIG_DIR=${CONFIG_DIR:-/home/joosep/particleflow/}
WORKSPACE_DIR=${WORKSPACE_DIR}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-data_val}
INPUT_FILELIST=${INPUT_FILELIST:-${CONFIG_DIR}/scripts/cmssw/jetmet0.txt}

# Derived paths
FILENAME=$(sed -n "${SEED}p" $INPUT_FILELIST)
OUTDIR=${WORKSPACE_DIR}/val_data/${JOB_TYPE}/${OUTPUT_SUBDIR}/${SAMPLE}

if [ -z "$FILENAME" ]; then
    echo "Error: Input file for line $SEED not found in $INPUT_FILELIST."
    exit 1
fi

if [ -z "$WORKDIR" ]; then
    WORKDIR=/scratch/local/joosep/val_data_${SAMPLE}_${SEED}_${JOB_TYPE}
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

# Data conditions and setup
CONDITIONS=140X_dataRun3_v20
ERA=Run3
GEOM=DB:Extended

if [ "$JOB_TYPE" == "mlpf" ]; then
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,PAT --scenario pp \
        --datatier RECO,MINIAOD -n -1 --era $ERA \
        --eventcontent RECO,MINIAOD --geometry=$GEOM \
        --nThreads $NTHREADS --filein "$FILENAME" --fileout file:step3.root \
        --procModifiers mlpf --no_exec --python_filename=step3.py --data
    echo "process.mlpfProducer.use_cuda = ${USE_CUDA}" >> step3.py
else
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,PAT --scenario pp \
        --datatier RECO,MINIAOD -n -1 --era $ERA \
        --eventcontent RECO,MINIAOD --geometry=$GEOM \
        --nThreads $NTHREADS --filein "$FILENAME" --fileout file:step3.root \
        --no_exec --python_filename=step3.py --data
fi

cmsRun step3.py

# NANO recipe for data
cmsDriver.py step4 -s NANO --data --conditions $CONDITIONS --era $ERA \
    --eventcontent NANOAOD --datatier NANOAOD --scenario pp \
    --customise_commands="process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)));process.MessageLogger.cerr.FwkReport.reportEvery=1000" \
    -n -1 --no_exec --filein file:step3_inMINIAOD.root --fileout file:step4_NANO.root --python_filename=step4.py

cmsRun step4.py

# Copy outputs
cp step4_NANO.root ${OUTDIR}/step4_NANO_${SEED}.root
