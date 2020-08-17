#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 4

cd ~/particleflow

#TF training
#singularity exec --nv ~/particleflow/mlpf python3 test/tf_model.py --datapath data/TTbar_14TeV_TuneCUETP8M1_cfi/tfr2 --target cand --ntrain 10000 --ntest 10000

#Pytorch training
singularity exec -B /scratch -B /home --nv /home/software/singularity/base.simg:2020-08-13 \
  python3 test/train_end2end.py \
  --dataset /home/joosep/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi \
  --n_train 4000 --n_val 1000 \
  --model PFNet7 --convlayer gravnet-radius --lr 0.00005 \
  --hidden_dim 256 --n_epochs 100 \
  --l1 1.0 --l2 0.001 --space_dim 3 \
  --target cand --batch_size 1 --activation leaky_relu \
  --dropout 0.3 --n_plot 20 --convlayer2 gatconv --load data/DataParallel_TTbar_14TeV_TuneCUETP8M1_cfi_cand__npar_980317__cfg_ca49e7d9e0__user_joosep__ntrain_4000__lr_5e-05__1597301424/epoch_180/weights.pth
