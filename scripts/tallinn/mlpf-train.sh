#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.13.0.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py train -c parameters/clic.yaml \
    --plot-freq 1 \
    --weights ../test/particleflow/models/mlpf-clic-2023-results/clusters_best_tuned_gnn_clic_v130/weights/weights-96-5.346523.hdf5 \
    --batch-multiplier 0.5
