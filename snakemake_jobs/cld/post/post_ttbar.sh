#!/bin/bash
set -e
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TMPDIR=/tmp/particleflow/tmp
export TEMPDIR=/tmp/particleflow/tmp
export TEMP=/tmp/particleflow/tmp
export TMP=/tmp/particleflow/tmp
mkdir -p $TMPDIR
cd /home/joosep/particleflow

export PYTHONPATH=$(pwd):$PYTHONPATH
start_seed=$1
for (( i=0; i<1; i++ )); do
    seed=$((start_seed + i))
    
    if [ ! -f /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/post/p8_ee_ttbar_ecm365/reco_p8_ee_ttbar_ecm365_${seed}.parquet ]; then
        if [ -f /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_ttbar_ecm365/root/reco_p8_ee_ttbar_ecm365_${seed}.root ]; then
            echo "Postprocessing /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_ttbar_ecm365/root/reco_p8_ee_ttbar_ecm365_${seed}.root"
            python3 mlpf/data/key4hep/postprocessing.py --input /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_ttbar_ecm365/root/reco_p8_ee_ttbar_ecm365_${seed}.root --outpath /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/post/p8_ee_ttbar_ecm365 --detector cld
            if [ -f /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/post/p8_ee_ttbar_ecm365/reco_p8_ee_ttbar_ecm365_${seed}.parquet ]; then
                python3 -c "import awkward as ak; ak.from_parquet('/mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/post/p8_ee_ttbar_ecm365/reco_p8_ee_ttbar_ecm365_${seed}.parquet')"
            else
                echo "Error: Postprocessing failed to produce /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/post/p8_ee_ttbar_ecm365/reco_p8_ee_ttbar_ecm365_${seed}.parquet"
                exit 1
            fi
        else
            echo "Error: Input file /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/gen/p8_ee_ttbar_ecm365/root/reco_p8_ee_ttbar_ecm365_${seed}.root missing for postprocessing"
            exit 1
        fi
    else
        echo "Skipping /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/post/p8_ee_ttbar_ecm365/reco_p8_ee_ttbar_ecm365_${seed}.parquet, already exists"
    fi

done
