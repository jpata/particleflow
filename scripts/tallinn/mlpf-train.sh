#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=8G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=docker://nvcr.io/nvidia/tensorflow:23.05-tf2-py3
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent -B /local --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/local/joosep/mlpf/tensorflow_datasets \
    --env TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" \
    $IMG python mlpf/pipeline.py train -c parameters/clic-hits.yaml --plot-freq 1 --num-cpus 16 --batch-multiplier 1 --ntrain 100000 --ntest 100000
