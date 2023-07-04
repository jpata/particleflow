#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.12.0-nvidia.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py train -c parameters/clic-hits-ln.yaml \
    --plot-freq 1 --num-cpus 32 --batch-multiplier 2 \
    --weights experiments/clic-hits-ln_20230626_123309_931116.gpu1.local/weights/weights-12-0.172574.hdf5

#    --env TF_GPU_THREAD_MODE=gpu_private \
#    --env TF_GPU_THREAD_COUNT=8 \
#    --env TF_XLA_FLAGS="--tf_xla_auto_jit=2" \
