#!/bin/bash
set -e
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/scratch/local/joosep/tmp/job_${SLURM_JOB_ID:-manual-$$}"
export TEMPDIR=$TMPDIR
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR
cd /home/joosep/particleflow

start_seed=$1
for (( i=0; i<1; i++ )); do
    seed=$((start_seed + i))
    export OUTDIR=/local/joosep/mlpf/cld/v1.4.0/50k/gen/ && export CONFIG_DIR=/home/joosep/particleflow/mlpf/data/key4hep/gen/cld/CLDConfig && export WORKDIR="/scratch/local/joosep/cld_p8_ee_WW_fullhad_ecm365_${seed}_${SLURM_JOB_ID:-manual-$$}" && export NEV=10 && export KEY4HEP_RELEASE=2026-04-08 && export LD_PRELOAD=/usr/lib64/libffi.so.8
    
    if [ ! -f /local/joosep/mlpf/cld/v1.4.0/50k/gen/p8_ee_WW_fullhad_ecm365/root/reco_p8_ee_WW_fullhad_ecm365_${seed}.root ]; then
        echo "Generating /local/joosep/mlpf/cld/v1.4.0/50k/gen/p8_ee_WW_fullhad_ecm365/root/reco_p8_ee_WW_fullhad_ecm365_${seed}.root"
        
        (set -o pipefail; bash mlpf/data/key4hep/gen/cld/run_sim.sh p8_ee_WW_fullhad_ecm365 $seed nopu 2>&1 | sed -u -E '/^[^[:space:]]+[[:space:]]+(DEBUG|VERBOSE)[[:space:]]/d')
    else
        echo "Skipping /local/joosep/mlpf/cld/v1.4.0/50k/gen/p8_ee_WW_fullhad_ecm365/root/reco_p8_ee_WW_fullhad_ecm365_${seed}.root, already exists"
    fi
    (source /cvmfs/sw.hsf.org/key4hep/setup.sh -r "${KEY4HEP_RELEASE:-${KEY4HEP_VERSION:-2026-04-08}}" && python3 -c 'import ROOT,sys; f=ROOT.TFile.Open(sys.argv[1]); assert f and not f.IsZombie(), sys.argv[1]; t=f.Get("events"); assert t and t.GetEntries()==10, sys.argv[1]' "/local/joosep/mlpf/cld/v1.4.0/50k/gen/p8_ee_WW_fullhad_ecm365/root/reco_p8_ee_WW_fullhad_ecm365_${seed}.root")
done
