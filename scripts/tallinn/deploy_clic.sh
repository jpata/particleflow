#!/bin/bash


rm -f files_to_copy.txt
maxfiles=100
path=experiments/pyg-clic_20250306_105311_290722/./preds_checkpoint-10-1.932789
targetpath=/scratch/persistent/joosep/huggingface/particleflow/clic/clusters/v2.3.0/pyg-clic_20250130_214007_333962/

mkdir -p $targetpath
cp $path/../* $targetpath/
cp -R $path/../checkpoints $targetpath/
cp -R $path/../history $targetpath/
cp -R $path/../runs $targetpath/

samplestocopy=(
    "clic_edm_qq_pf"
    "clic_edm_ttbar_pf"
    "clic_edm_ww_fullhad_pf"
)

for sample in "${samplestocopy[@]}"; do
    find "$path/$sample" -type f | sort | head -n$maxfiles >> files_to_copy.txt
done

rsync --progress --relative --files-from files_to_copy.txt `pwd` $targetpath
