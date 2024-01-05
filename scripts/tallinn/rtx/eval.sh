#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.14.0.simg
#IMG=/home/joosep/singularity/tf-2.14.0.simg
cd ~/particleflow2

#change these
EXPDIR=experiments/mlpf-clic-2023-results/clusters_best_tuned_gnn_clic_v130/
WEIGHTS=experiments/mlpf-clic-2023-results/clusters_best_tuned_gnn_clic_v130/weights/weights-96-5.346523.hdf5

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python3.10 mlpf/pipeline.py evaluate \
    --train-dir $EXPDIR --weights $WEIGHTS

#singularity exec -B /scratch/persistent --nv \
#    --env PYTHONPATH=hep_tfds \
#    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
#    $IMG python mlpf/pipeline.py plots \
#    --train-dir $EXPDIR
