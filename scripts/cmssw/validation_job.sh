#!/bin/bash
USE_CUDA=$1
JOBTYPE=$2
INPUT_FILELIST=$3
SAMPLE=$4
NJOB=$5

PREVDIR=`pwd`

#change this as needed, need enough space for outputs
OUTDIR=$CMSSW_BASE/out/
WORKDIR=$CMSSW_BASE/work_${SAMPLE}_${JOBTYPE}_${NJOB}

# uncomment the following when running at T2_EE_Estonia
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /scratch/persistent/joosep/CMSSW_15_0_5
eval `scram runtime -sh`
cd $PREVDIR
export OUTDIR=/local/$USER/mlpf/results/cms/${CMSSW_VERSION}_mlpf_v2.6.0pre1_puppi_2372e2/
export WORKDIR=/scratch/local/$USER/${SLURM_JOB_ID}

#abort on error, print all commands
set -e
set -x

CONDITIONS=auto:phase1_2023_realistic ERA=Run3 GEOM=DB.Extended CUSTOM=
FILENAME=`sed -n "${NJOB}p" $INPUT_FILELIST`
NTHREADS=4
NEV=-1

mkdir -p $WORKDIR
cd $WORKDIR

env

if [ $JOBTYPE == "mlpf" ]; then
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT \
	--datatier RECOSIM,MINIAODSIM --nThreads $NTHREADS -n $NEV --era $ERA \
	--eventcontent RECOSIM,MINIAODSIM --geometry=$GEOM \
	--filein $FILENAME --fileout file:step3.root --procModifiers mlpf --no_exec
    echo "process.mlpfProducer.use_cuda = ${USE_CUDA}" >> step3_RAW2DIGI_L1Reco_RECO_RECOSIM_PAT.py
    #echo "process.puppi.applyMLPF = False" >> step3_RAW2DIGI_L1Reco_RECO_RECOSIM_PAT.py
    echo "process.mlpfProducer.model_path = 'RecoParticleFlow/PFProducer/data/mlpf/mlpf_5M_attn2x3x256_bm12_relu_checkpoint10_8xmi250_fp32_fused_20250722.onnx'" >> step3_RAW2DIGI_L1Reco_RECO_RECOSIM_PAT.py
elif [ $JOBTYPE == "mlpfpu" ]; then
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT \
	--datatier RECOSIM,MINIAODSIM --nThreads $NTHREADS -n $NEV --era $ERA \
	--eventcontent RECOSIM,MINIAODSIM --geometry=$GEOM \
	--filein $FILENAME --fileout file:step3.root --procModifiers mlpf --no_exec
    echo "process.mlpfProducer.use_cuda = ${USE_CUDA}" >> step3_RAW2DIGI_L1Reco_RECO_RECOSIM_PAT.py
    #echo "process.puppi.applyMLPF = True" >> step3_RAW2DIGI_L1Reco_RECO_RECOSIM_PAT.py
    echo "process.mlpfProducer.model_path = 'RecoParticleFlow/PFProducer/data/mlpf/mlpf_5M_attn2x3x256_bm12_relu_checkpoint10_8xmi250_fp32_fused_20250722.onnx'" >> step3_RAW2DIGI_L1Reco_RECO_RECOSIM_PAT.py
elif [ $JOBTYPE == "pf" ]; then
    cmsDriver.py step3 --conditions $CONDITIONS \
        -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT \
	--datatier RECOSIM,MINIAODSIM --nThreads $NTHREADS -n $NEV --era $ERA \
	--eventcontent RECOSIM,MINIAODSIM --geometry=$GEOM \
	--filein $FILENAME --fileout file:step3.root --no_exec
fi

# echo """
# process.Timing = cms.Service(\"Timing\",
#     summaryOnly = cms.untracked.bool(False),
#     useJobReport = cms.untracked.bool(True)
# )""" >> step3_RAW2DIGI_L1Reco_RECO_RECOSIM_PAT.py

cmsRun step3_RAW2DIGI_L1Reco_RECO_RECOSIM_PAT.py

#Plain NANO recipe
#cmsDriver.py step4 -s NANO --mc --conditions $CONDITIONS --era $ERA \
#    --eventcontent NANOAODSIM --datatier NANOAODSIM \
#    --customise_commands="process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)));process.MessageLogger.cerr.FwkReport.reportEvery=1000" \
#    -n -1 --no_exec --filein file:step3_inMINIAODSIM.root --fileout file:step4_NANO.root
#BTV/PF NANO recipe
cmsDriver.py step4_btv -s NANO:@BTV --mc --conditions $CONDITIONS --era $ERA \
    --eventcontent NANOAODSIM --datatier NANOAODSIM \
    --customise_commands="process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)));process.MessageLogger.cerr.FwkReport.reportEvery=1000" \
    -n -1 --no_exec --filein file:step3_inMINIAODSIM.root --fileout file:step4_NANO_btv.root
#JME NANO recipe
cmsDriver.py step4_jme -s NANO:@JME --mc --conditions $CONDITIONS --era $ERA \
    --eventcontent NANOAODSIM --datatier NANOAODSIM \
    --customise_commands="process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)));process.MessageLogger.cerr.FwkReport.reportEvery=1000" \
    -n -1 --no_exec --filein file:step3_inMINIAODSIM.root --fileout file:step4_NANO_jme.root

#cmsRun step4_NANO.py
#cmsRun step4_btv_NANO.py
cmsRun step4_jme_NANO.py

ls *.root

mkdir -p $OUTDIR/cuda_${USE_CUDA}/${SAMPLE}_${JOBTYPE}

# cp step3.root $OUTDIR/cuda_${USE_CUDA}/${SAMPLE}_${JOBTYPE}/step3_RECO_${NJOB}.root
#cp step3_inMINIAODSIM.root $OUTDIR/cuda_${USE_CUDA}/${SAMPLE}_${JOBTYPE}/step3_MINI_${NJOB}.root
# cp step4_NANO.root $OUTDIR/${SAMPLE}_${JOBTYPE}/step4_NANO_${NJOB}.root
#cp step4_NANO_btv.root $OUTDIR/cuda_${USE_CUDA}/${SAMPLE}_${JOBTYPE}/step4_NANO_btv_${NJOB}.root
cp step4_NANO_jme.root $OUTDIR/cuda_${USE_CUDA}/${SAMPLE}_${JOBTYPE}/step4_NANO_jme_${NJOB}.root

#python3 ~/particleflow/mlpf/plotting/cms_fwlite.py step3_inMINIAODSIM.root step3_inMINIAODSIM.pkl
#cp step3_inMINIAODSIM.pkl $OUTDIR/cuda_${USE_CUDA}/${SAMPLE}_${JOBTYPE}/step3_MINI_${NJOB}.pkl

rm -Rf $WORKDIR
