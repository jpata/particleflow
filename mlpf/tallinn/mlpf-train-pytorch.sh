#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=10G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg
cd ~/particleflow/mlpf

#TF training
singularity exec -B /scratch-persistent --nv $IMG \
  python3 ssl_pipeline.py --data_split_mode mix \
  --prefix_VICReg pytorch_${SLURM_JOB_ID} --prefix_mlpf MLPF_test \
  --train_mlpf --native --n_epochs_VICReg 0 --bs_mlpf 100 \
  --n_epochs_mlpf 5000 --patience 50 --width_mlpf 256 --embedding_dim_mlpf 256 --lr 0.00005 --num_convs_mlpf 3 --nearest 32 --evaluate_mlpf
