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
cd /home/joosep/particleflow-dev

export PYTHONPATH=$(pwd):$PYTHONPATH
start_seed=$1
for (( i=0; i<1; i++ )); do
    seed=$((start_seed + i))
    if [ ! -f /local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/post/gun_e_10gev/reco_gun_e_10gev_${seed}.parquet ]; then
        if [ -f /local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/gen/gun_e_10gev/root/reco_gun_e_10gev_${seed}.root ]; then
            echo "Postprocessing /local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/gen/gun_e_10gev/root/reco_gun_e_10gev_${seed}.root"
            python3 mlpf/data/key4hep/postprocessing.py --input /local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/gen/gun_e_10gev/root/reco_gun_e_10gev_${seed}.root --outpath /local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/post/gun_e_10gev --detector idea
            if [ -f /local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/post/gun_e_10gev/reco_gun_e_10gev_${seed}.parquet ]; then
                python3 -c "import awkward as ak; ak.from_parquet('/local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/post/gun_e_10gev/reco_gun_e_10gev_${seed}.parquet')"
            else
                echo "Error: Postprocessing failed to produce /local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/post/gun_e_10gev/reco_gun_e_10gev_${seed}.parquet"
                exit 1
            fi
        else
            echo "Error: Input file /local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/gen/gun_e_10gev/root/reco_gun_e_10gev_${seed}.root missing for postprocessing"
            exit 1
        fi
    else
        echo "Skipping /local/joosep/mlpf/idea/IDEA_o1_v03_fccconfig_a05a3a9/post/gun_e_10gev/reco_gun_e_10gev_${seed}.parquet, already exists"
    fi

done
