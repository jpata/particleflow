#!/bin/bash
set -e
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TMPDIR=/scratch/local/joosep/tmp
export TEMPDIR=/scratch/local/joosep/tmp
export TEMP=/scratch/local/joosep/tmp
export TMP=/scratch/local/joosep/tmp
mkdir -p $TMPDIR
cd /home/joosep/particleflow

export PYTHONPATH=$(pwd):$PYTHONPATH
start_seed=$1
for (( i=0; i<1; i++ )); do
    seed=$((start_seed + i))
    
    if [ ! -f /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/post/pu55to75/TTbar_13p6TeV_TuneCUETP8M1_cfi/pfntuple_${seed}.pkl.bz2 ]; then
        if [ -f /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/gen/pu55to75/TTbar_13p6TeV_TuneCUETP8M1_cfi/root/pfntuple_${seed}.root ]; then
            echo "Postprocessing /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/gen/pu55to75/TTbar_13p6TeV_TuneCUETP8M1_cfi/root/pfntuple_${seed}.root"
            python3 mlpf/data/cms/postprocessing2.py --input /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/gen/pu55to75/TTbar_13p6TeV_TuneCUETP8M1_cfi/root/pfntuple_${seed}.root --outpath /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/post/pu55to75/TTbar_13p6TeV_TuneCUETP8M1_cfi --num_events -1
            if [ -f /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/post/pu55to75/TTbar_13p6TeV_TuneCUETP8M1_cfi/pfntuple_${seed}.pkl ]; then
                bzip2 -z /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/post/pu55to75/TTbar_13p6TeV_TuneCUETP8M1_cfi/pfntuple_${seed}.pkl
            else
                echo "Error: Postprocessing failed to produce /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/post/pu55to75/TTbar_13p6TeV_TuneCUETP8M1_cfi/pfntuple_${seed}.pkl"
                exit 1
            fi
        else
            echo "Error: Input file /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/gen/pu55to75/TTbar_13p6TeV_TuneCUETP8M1_cfi/root/pfntuple_${seed}.root missing for postprocessing"
            exit 1
        fi
    else
        echo "Skipping /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/post/pu55to75/TTbar_13p6TeV_TuneCUETP8M1_cfi/pfntuple_${seed}.pkl.bz2, already exists"
    fi

done
