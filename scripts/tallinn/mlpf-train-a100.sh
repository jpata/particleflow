#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:2
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.13.0.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py train -c parameters/clic-hits.yaml \
    --plot-freq 1 --num-cpus 32 --batch-multiplier 2 \
    --weights experiments/clic-hits_20230804_231819_093246.gpu1.local/weights/weights-04-0.195574.hdf5

#    --env TF_GPU_THREAD_MODE=gpu_private \
#    --env TF_GPU_THREAD_COUNT=8 \
#    --env TF_XLA_FLAGS="--tf_xla_auto_jit=2" \
