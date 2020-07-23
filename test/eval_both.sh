#!/bin/bash
#Evaluation of both models producing all the plots in two different folders.
# $1 should be the output file from training PFNet
# $2 should be the output file from training PFNet2

IMG=~jpata/gpuservers/singularity/images/pytorch.simg


singularity exec -B /storage --nv $IMG \
  python3 test/pred_tf_model.py --nhidden 256 --distance-dim 256 --convlayer ghconv --num-conv 4 --ntrain 400000 --ntest 100000 --weights $1 --gpu --datapath /storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi/tfr3 --model PFNet --target cand \
&& \
singularity exec -B /storage $IMG \
 python3 test/plots.py --pkl df_1.pkl.bz2 --target cand \
&& \
singularity exec -B /storage --nv $IMG \
  python3 test/pred_tf_model.py --nhidden 256 --distance-dim 256 --convlayer ghconv --num-conv 4 --ntrain 400000 --ntest 100000 --weights $2 --gpu --datapath /storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi/tfr3 --model PFNet2 --target cand 

singularity exec -B /storage $IMG \
 python3 test/plots.py --pkl df_1.pkl.bz2 --target cand
 
