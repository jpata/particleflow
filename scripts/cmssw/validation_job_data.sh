#!/bin/bash
JOBTYPE=$1
INPUT_FILELIST=$2
SAMPLE=$3
NJOB=$4

PREVDIR=`pwd`

#change this as needed, need enough space for outputs
OUTDIR=$CMSSW_BASE/out/
WORKDIR=$CMSSW_BASE/work_${SAMPLE}_${JOBTYPE}_${NJOB}

# uncomment the following when running at T2_EE_Estonia
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /scratch/persistent/joosep/CMSSW_15_0_1
eval `scram runtime -sh`
cd $PREVDIR
export OUTDIR=/scratch/local/$USER/mlpf/results/cms/${CMSSW_VERSION}/
export WORKDIR=/scratch/local/$USER/${SLURM_JOB_ID}

#abort on error, print all commands
set -e
set -x

CONDITIONS=140X_dataRun3_Prompt_v2 ERA=Run3 GEOM=DB.Extended CUSTOM=
FILENAME=`sed -n "${NJOB}p" $INPUT_FILELIST`
NTHREADS=8

mkdir -p $WORKDIR
cd $WORKDIR

env

if [ $JOBTYPE == "mlpf" ]; then
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,PAT --scenario pp \
	--datatier RECO,MINIAOD --nThreads $NTHREADS -n -1 --era $ERA \
	--eventcontent RECO,MINIAOD --geometry=$GEOM \
	--filein $FILENAME --fileout file:step3.root --procModifiers mlpf --no_exec --data
    echo "process.mlpfProducer.use_cuda = True" >> step3_RAW2DIGI_L1Reco_RECO_PAT.py
elif [ $JOBTYPE == "pf" ]; then
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,PAT --scenario pp \
	--datatier RECO,MINIAOD --nThreads $NTHREADS -n -1 --era $ERA \
	--eventcontent RECO,MINIAOD --geometry=$GEOM \
	--filein $FILENAME --fileout file:step3.root --no_exec --data
fi

echo """
process.Timing = cms.Service(\"Timing\",
    summaryOnly = cms.untracked.bool(False),
    useJobReport = cms.untracked.bool(True)
)""" >> step3_RAW2DIGI_L1Reco_RECO_PAT.py

cmsRun step3_RAW2DIGI_L1Reco_RECO_PAT.py

cmsDriver.py step4 -s NANO --data --conditions $CONDITIONS --era $ERA \
    --eventcontent NANOAOD --datatier NANOAOD --data --scenario pp \
    --customise_commands="process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)));process.MessageLogger.cerr.FwkReport.reportEvery=1000" \
     -n -1 --no_exec --filein file:step3_inMINIAOD.root --fileout file:step4_NANO.root

cmsRun step4_NANO.py

ls *.root

mkdir -p $OUTDIR/${SAMPLE}_${JOBTYPE}

cp step3.root $OUTDIR/${SAMPLE}_${JOBTYPE}/step3_RECO_${NJOB}.root
cp step3_inMINIAOD.root $OUTDIR/${SAMPLE}_${JOBTYPE}/step3_MINI_${NJOB}.root
cp step4_NANO.root $OUTDIR/${SAMPLE}_${JOBTYPE}/step4_NANO_${NJOB}.root

rm -Rf $WORKDIR
