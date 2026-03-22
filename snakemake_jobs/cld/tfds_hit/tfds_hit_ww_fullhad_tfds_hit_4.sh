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
export KERAS_BACKEND=torch
hostname
export TFDS_VERSION=3.1.0
env

echo "Building TFDS for mlpf/heptfds/cld_pf_edm4hep_hits/ww_fullhad config 4"
echo "Manual dir: /local/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29/post"
echo "Scratch dir: /scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4"

mkdir -p /scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4
cleanup() {
    if [ ! -z "/scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4" ] && [ "/scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4" != "/scratch/local/joosep" ]; then
        echo "Cleaning up scratch directory /scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4"
        rm -Rf /scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4
    fi
}
trap cleanup EXIT
export TMPDIR=/scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4
export TEMPDIR=/scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4
export TEMP=/scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4
export TMP=/scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4
tfds build mlpf/heptfds/cld_pf_edm4hep_hits/ww_fullhad --config 4 --data_dir /scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4 --manual_dir /local/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29/post --overwrite
echo "Copying from /scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4 to /local/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds"
cp -r /scratch/local/joosep/tfds_tmp/ww_fullhad_tfds_hit_4/* /local/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds/
