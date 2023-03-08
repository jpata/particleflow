#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=10G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg
cd ~/particleflow

#pytorch training
singularity exec -B /scratch-persistent --nv $IMG \
  python3 mlpf/pyg_pipeline.py --dataset CLIC --data_path data/clic_edm4hep/ \
  --outpath experiments/pytorch_${SLURM_JOB_ID} --bs 200 --n_train 1800 \
  --n_valid 100 --n_test 100 --n_epochs 500 --propagate_dim 8 \
  --space_dim 4 --nearest 32 --embedding_dim 256
