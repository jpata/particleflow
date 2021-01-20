#!/bin/bash

IMG=~jpata/gpuservers/singularity/images/pytorch.simg

singularity exec -B /storage --nv $IMG \
  python3 test/pred_tf_model.py --nhidden 256 --distance-dim 256 --convlayer ghconv --num-conv 4 --ntrain 400000 --ntest 100000 --weights $1 --gpu

#singularity exec -B /storage $IMG \
#  python3 test/plots.py --pkl df_1.pkl.bz2 --target cand
