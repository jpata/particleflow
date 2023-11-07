#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu=40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.14.0.simg
cd ~/particleflow

# export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python3.10 mlpf/pipeline.py train -c parameters/cms-gen.yaml --plot-freq 1 --num-cpus 32 \
    --batch-multiplier 8 --weights experiments/cms-gen_20231104_082738_031639.gpu1.local/weights/weights-10-4.203166.hdf5
