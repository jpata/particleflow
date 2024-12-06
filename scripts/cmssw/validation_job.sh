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
cd /scratch/persistent/joosep/CMSSW_14_1_0_pre3
eval `scram runtime -sh`
cd $PREVDIR
export OUTDIR=/local/joosep/mlpf/results/cms/${CMSSW_VERSION}_fcd442/
export WORKDIR=/scratch/local/$USER/${SLURM_JOB_ID}

#abort on error, print all commands
set -e
set -x

CONDITIONS=auto:phase1_2023_realistic ERA=Run3 GEOM=DB.Extended CUSTOM=
FILENAME=`sed -n "${NJOB}p" $INPUT_FILELIST`
NTHREADS=1

mkdir -p $WORKDIR
cd $WORKDIR

env

if [ $JOBTYPE == "mlpf" ]; then
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT \
	--datatier RECOSIM,MINIAODSIM --nThreads 1 -n -1 --era $ERA \
	--eventcontent RECOSIM,MINIAODSIM --geometry=$GEOM \
	--filein $FILENAME --fileout file:step3.root --procModifiers mlpf
elif [ $JOBTYPE == "pf" ]; then
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT \
	--datatier RECOSIM,MINIAODSIM --nThreads 1 -n -1 --era $ERA \
	--eventcontent RECOSIM,MINIAODSIM --geometry=$GEOM \
	--filein $FILENAME --fileout file:step3.root
fi

#JME NANO recipe
cmsDriver.py step3 -s NANO --mc --conditions $CONDITIONS --era $ERA \
    --eventcontent NANOAODSIM --datatier NANOAODSIM \
    --customise_commands="process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)));process.MessageLogger.cerr.FwkReport.reportEvery=1000" \
    -n -1 --no_exec --filein file:step3_inMINIAODSIM.root --fileout file:step3_NANO.root

echo "from PhysicsTools.NanoAOD.custom_jme_cff import PrepJMECustomNanoAOD" >> step3_NANO.py
echo "process = PrepJMECustomNanoAOD(process)" >> step3_NANO.py
cmsRun step3_NANO.py

ls *.root

mkdir -p $OUTDIR/${SAMPLE}_${JOBTYPE}

#convert CMSSW EDM to pkl for easy plotting
python3 $PREVDIR/mlpf/plotting/cms_fwlite.py step3_inMINIAODSIM.root step3.pkl

cp step3.root $OUTDIR/${SAMPLE}_${JOBTYPE}/step3_RECO_${NJOB}.root
cp step3_inMINIAODSIM.root $OUTDIR/${SAMPLE}_${JOBTYPE}/step3_MINI_${NJOB}.root
cp step3_NANO.root $OUTDIR/${SAMPLE}_${JOBTYPE}/step3_NANO_${NJOB}.root
cp step3.pkl $OUTDIR/${SAMPLE}_${JOBTYPE}/step3_MINI_${NJOB}.pkl

rm -Rf $WORKDIR
