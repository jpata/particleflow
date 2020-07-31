#!/bin/bash
# Train both PFNet and PFNet2 models one after the other
# Parameters have been set to be exactly the same for both models

singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
  python3 test/tf_model.py --target cand --ntrain 400000 --ntest 100000 --nepochs 25 --lr 0.00001 \
  --datapath /storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi \
  --nhidden 256 --distance-dim 256 --num-conv 4 --weights inverse --lr-decay 0 --convlayer ghconv --model PFNet \
&& \
singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
  python3 test/tf_model.py --target cand --ntrain 400000 --ntest 100000 --nepochs 25 --lr 0.00001 \
  --datapath /storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi \
  --nhidden 256 --distance-dim 256 --num-conv 4 --weights inverse --lr-decay 0 --convlayer ghconv --model PFNet2
  
