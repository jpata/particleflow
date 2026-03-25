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
cd /home/joosep/particleflow2

config_id=$1
tfds_id=ttbar_nopu_tfds_$config_id
job_scratch_dir=/scratch/local/joosep/tfds_tmp/$tfds_id

export PYTHONPATH=$(pwd):$PYTHONPATH
export KERAS_BACKEND=torch
hostname
export TFDS_VERSION=3.0.0
env

echo "Building TFDS for mlpf/heptfds/cms_pf/ttbar_nopu config $config_id"
echo "Manual dir: /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/post/nopu"
echo "Scratch dir: $job_scratch_dir"

mkdir -p $job_scratch_dir
cleanup() {
    if [ ! -z "$job_scratch_dir" ] && [ "$job_scratch_dir" != "/scratch/local/joosep" ]; then
        echo "Cleaning up scratch directory $job_scratch_dir"
        rm -Rf $job_scratch_dir
    fi
}
trap cleanup EXIT
export TMPDIR=$job_scratch_dir
export TEMPDIR=$job_scratch_dir
export TEMP=$job_scratch_dir
export TMP=$job_scratch_dir
tfds build mlpf/heptfds/cms_pf/ttbar_nopu --config $config_id --data_dir $job_scratch_dir --manual_dir /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/post/nopu --overwrite
echo "Copying from $job_scratch_dir to /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/tfds"
cp -r $job_scratch_dir/* /local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/tfds/
