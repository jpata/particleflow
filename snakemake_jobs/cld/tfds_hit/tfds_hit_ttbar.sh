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

config_id=$1
tfds_id=ttbar_tfds_hit_$config_id
job_scratch_dir=/tmp/particleflow/tfds_tmp/$tfds_id

export PYTHONPATH=$(pwd):$PYTHONPATH
export KERAS_BACKEND=torch
hostname
export TFDS_VERSION=3.2.0
env

echo "Building TFDS for mlpf/heptfds/cld_pf_edm4hep_hits/ttbar config $config_id"
echo "Manual dir: /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/post"
echo "Scratch dir: $job_scratch_dir"

mkdir -p $job_scratch_dir
cleanup() {
    if [ ! -z "$job_scratch_dir" ] && [ "$job_scratch_dir" != "/tmp/particleflow" ]; then
        echo "Cleaning up scratch directory $job_scratch_dir"
        rm -Rf $job_scratch_dir
    fi
}
trap cleanup EXIT
export TMPDIR=$job_scratch_dir
export TEMPDIR=$job_scratch_dir
export TEMP=$job_scratch_dir
export TMP=$job_scratch_dir
tfds build mlpf/heptfds/cld_pf_edm4hep_hits/ttbar --config $config_id --data_dir $job_scratch_dir --manual_dir /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/post --overwrite
echo "Copying from $job_scratch_dir to /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/tfds"
cp -r $job_scratch_dir/* /mnt/work/mlpf//cld/v1.2.5_key4hep_2025-05-29/tfds/
