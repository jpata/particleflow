#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:2
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.14.0.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG horovodrun -np 2 -H localhost:2 python3.10 mlpf/pipeline.py train -c parameters/clic-test.yaml \
    --plot-freq 0 --num-cpus 32 --batch-multiplier 5 \
    --horovod-enabled --ntrain 50000 --ntest 50000 --nepochs 11 --benchmark_dir exp_dir

#    --env TF_GPU_THREAD_MODE=gpu_private \
#    --env TF_GPU_THREAD_COUNT=8 \
#    --env TF_XLA_FLAGS="--tf_xla_auto_jit=2" \
