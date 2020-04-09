#!/bin/bash

IMG=~jpata/gpuservers/singularity/images/pytorch.simg

DATASET=TTbar_14TeV_TuneCUETP8M1_cfi
PERFILE=1
MERGE=1

#DATASET=SingleElectronFlatPt1To100_pythia8_cfi
#PERFILE=-1
#MERGE=5

#DATASET=SinglePiFlatPt0p7To10_cfi
#PERFILE=-1
#MERGE=10

cd test/$DATASET
rm -Rf raw processed
mkdir -p raw
mkdir -p processed
\ls -1 pfntuple*.root | parallel --gnu -j24 python ../postprocessing2.py --input {} --events-per-file $PERFILE

cd ../..
singularity exec --nv $IMG python test/graph_data.py --dataset test/$DATASET --num-files-merge $MERGE

singularity exec --nv $IMG python test/train_end2end.py --lr 0.0001 --n_train 9000 --n_test 1000 --hidden_dim 512 --target gen --model PFNet7 --n_epochs 100 --dropout 0.2 --n_plot 10 --dataset test/TTbar_14TeV_TuneCUETP8M1_cfi --l3 0
