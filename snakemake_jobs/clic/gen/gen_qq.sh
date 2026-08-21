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

start_seed=$1
for (( i=0; i<1; i++ )); do
    seed=$((start_seed + i))
    if [ ! -f /local/joosep/mlpf/clic/v1.2.5_key4hep_2025-05-29/gen/p8_ee_qq_ecm380/root/reco_p8_ee_qq_ecm380_${seed}.root ]; then
        echo "Generating /local/joosep/mlpf/clic/v1.2.5_key4hep_2025-05-29/gen/p8_ee_qq_ecm380/root/reco_p8_ee_qq_ecm380_${seed}.root"
        export OUTDIR=/local/joosep/mlpf/clic/v1.2.5_key4hep_2025-05-29/gen/ && export CONFIG_DIR=/home/joosep/particleflow/mlpf/data/key4hep/gen/clic && export WORKDIR=/scratch/local/joosep/p8_ee_qq_ecm380_$seed && export NEV=100
        bash mlpf/data/key4hep/gen/clic/run_sim.sh p8_ee_qq_ecm380 $seed nopu
    else
        echo "Skipping /local/joosep/mlpf/clic/v1.2.5_key4hep_2025-05-29/gen/p8_ee_qq_ecm380/root/reco_p8_ee_qq_ecm380_${seed}.root, already exists"
    fi
done
