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

config_id=$1
tfds_id=cld_qq_tfds_hit_$config_id
job_scratch_dir=/scratch/local/joosep/tfds_tmp/${tfds_id}_${SLURM_JOB_ID:-manual-$$}

export PYTHONPATH=$(pwd):$PYTHONPATH
export KERAS_BACKEND=torch
hostname
export TFDS_VERSION=3.3.0
env

echo "Building TFDS for mlpf/heptfds/cld_pf_edm4hep_hits/qq config $config_id"
echo "Manual dir: /local/joosep/mlpf/cld/v1.4.0/50k/post"
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
tfds build mlpf/heptfds/cld_pf_edm4hep_hits/qq --config $config_id --data_dir $job_scratch_dir --manual_dir /local/joosep/mlpf/cld/v1.4.0/50k/post --overwrite
echo "Copying from $job_scratch_dir to /local/joosep/mlpf/cld/v1.4.0/50k/tfds"
python3 -c 'import shutil,sys; shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True)' "$job_scratch_dir" "/local/joosep/mlpf/cld/v1.4.0/50k/tfds"
