#!/bin/bash
set -e
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TMPDIR="/scratch/local/joosep/tmp/job_${SLURM_JOB_ID:-manual-$$}"
export TEMPDIR=$TMPDIR
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR
cd /home/joosep/particleflow

export PYTHONPATH=$(pwd):$PYTHONPATH
start_seed=$1
for (( i=0; i<1; i++ )); do
    seed=$((start_seed + i))
    if [ ! -f /local/joosep/mlpf/ild/v1.4.0/50k/post/p8_ee_Z_qq_ecm91/reco_p8_ee_Z_qq_ecm91_${seed}.parquet ]; then
        if [ -f /local/joosep/mlpf/ild/v1.4.0/50k/gen/p8_ee_Z_qq_ecm91/root/reco_p8_ee_Z_qq_ecm91_${seed}.root ]; then
            echo "Postprocessing /local/joosep/mlpf/ild/v1.4.0/50k/gen/p8_ee_Z_qq_ecm91/root/reco_p8_ee_Z_qq_ecm91_${seed}.root"
            python3 mlpf/data/key4hep/postprocessing.py --input /local/joosep/mlpf/ild/v1.4.0/50k/gen/p8_ee_Z_qq_ecm91/root/reco_p8_ee_Z_qq_ecm91_${seed}.root --outpath /local/joosep/mlpf/ild/v1.4.0/50k/post/p8_ee_Z_qq_ecm91 --detector ild
            if [ -f /local/joosep/mlpf/ild/v1.4.0/50k/post/p8_ee_Z_qq_ecm91/reco_p8_ee_Z_qq_ecm91_${seed}.parquet ]; then
                python3 -c "import awkward as ak; ak.from_parquet('/local/joosep/mlpf/ild/v1.4.0/50k/post/p8_ee_Z_qq_ecm91/reco_p8_ee_Z_qq_ecm91_${seed}.parquet')"
            else
                echo "Error: Postprocessing failed to produce /local/joosep/mlpf/ild/v1.4.0/50k/post/p8_ee_Z_qq_ecm91/reco_p8_ee_Z_qq_ecm91_${seed}.parquet"
                exit 1
            fi
        else
            echo "Error: Input file /local/joosep/mlpf/ild/v1.4.0/50k/gen/p8_ee_Z_qq_ecm91/root/reco_p8_ee_Z_qq_ecm91_${seed}.root missing for postprocessing"
            exit 1
        fi
    else
        echo "Skipping /local/joosep/mlpf/ild/v1.4.0/50k/post/p8_ee_Z_qq_ecm91/reco_p8_ee_Z_qq_ecm91_${seed}.parquet, already exists"
    fi

done
